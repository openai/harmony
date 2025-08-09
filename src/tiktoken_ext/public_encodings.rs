use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read as _, Write as _},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use base64::{prelude::BASE64_STANDARD, Engine as _};

use crate::tiktoken::{CoreBPE, Rank};
use fs2::FileExt;
use sha1::Sha1;
use sha2::{Digest as _, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("the env var TIKTOKEN_ENCODINGS_BASE is not set, or invalid")]
    InvalidEncodingBaseDirEnvVar,

    #[error("unknown encoding name: {0}")]
    UnknownEncodingName(String),

    #[error("invalid tiktoken vocab file: {0}")]
    InvalidTiktokenVocabFile(#[source] std::io::Error),

    #[error("failed to create CoreBPE: {0}")]
    CoreBPECreationFailed(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("error downloading or loading vocab file: {0}")]
    DownloadOrLoadVocabFile(
        #[source]
        #[from]
        RemoteVocabFileError,
    ),

    #[error("failed to extend encoding")]
    FailedToExtendEncoding(#[source] Box<dyn std::error::Error + Send + Sync>),
}

#[derive(Debug, thiserror::Error)]
pub enum RemoteVocabFileError {
    #[error("failed to download or load vocab file")]
    FailedToDownloadOrLoadVocabFile(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("an underlying IO error occurred while {0}: {1}")]
    IOError(String, #[source] std::io::Error),

    #[error("hash mismatch for remote file {file_url}")]
    HashMismatch {
        file_url: String,
        expected_hash: String,
        computed_hash: String,
    },
}

const TIKTOKEN_ENCODINGS_BASE_VAR: &str = "TIKTOKEN_ENCODINGS_BASE";
const DEFAULT_TIKTOKEN_BASE_URL: &str = "https://openaipublic.blob.core.windows.net/encodings/";

static TIKTOKEN_BASE_URL_OVERRIDE: OnceLock<String> = OnceLock::new();

pub fn set_tiktoken_base_url(base_url: impl Into<String>) {
    let mut base = base_url.into();
    if !base.ends_with('/') {
        base.push('/');
    }
    // ignore error if already set
    let _ = TIKTOKEN_BASE_URL_OVERRIDE.set(base);
}

fn tiktoken_base_url() -> &'static str {
    TIKTOKEN_BASE_URL_OVERRIDE
        .get()
        .map(|s| s.as_str())
        .unwrap_or(DEFAULT_TIKTOKEN_BASE_URL)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    O200kBase,
    O200kHarmony,
    Cl100kBase,
}

impl Encoding {
    pub fn all() -> &'static [Self] {
        &[Self::O200kBase, Self::O200kHarmony, Self::Cl100kBase]
    }

    pub fn from_name(name: impl AsRef<str>) -> Option<Self> {
        let name_str = name.as_ref();
        for encoding in Self::all() {
            if encoding.name() == name_str {
                return Some(*encoding);
            }
        }
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_name(name: impl AsRef<str>) -> Result<CoreBPE, LoadError> {
        let name = name.as_ref();
        Self::from_name(name)
            .ok_or_else(|| LoadError::UnknownEncodingName(name.to_string()))?
            .load()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load_from_name(name: impl AsRef<str>) -> Result<CoreBPE, LoadError> {
        let name = name.as_ref();
        Self::from_name(name)
            .ok_or_else(|| LoadError::UnknownEncodingName(name.to_string()))?
            .load()
            .await
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base",
            Self::O200kHarmony => "o200k_harmony",
            Self::Cl100kBase => "cl100k_base",
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(&self) -> Result<CoreBPE, LoadError> {
        #[cfg(not(target_arch = "wasm32"))]
        let (vocab_file_path, check_hash) =
            if let Ok(base_dir) = std::env::var(TIKTOKEN_ENCODINGS_BASE_VAR) {
                (PathBuf::from(base_dir).join(self.vocab_file_name()), true)
            } else {
                let url = self.public_vocab_file_url();
                (
                    download_or_find_cached_file(&url, Some(self.expected_hash()))
                        .map_err(LoadError::DownloadOrLoadVocabFile)?,
                    false,
                )
            };

        match self {
            Self::O200kHarmony => {
                let mut specials: Vec<(String, Rank)> = self
                    .special_tokens()
                    .iter()
                    .map(|(s, r)| ((*s).to_string(), *r))
                    .collect();
                specials.extend((200014..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
                #[cfg(not(target_arch = "wasm32"))]
                {
                    load_encoding_from_file(
                        vocab_file_path,
                        check_hash.then(|| self.expected_hash()),
                        specials,
                        &self.pattern(),
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
                }
            }
            Self::O200kBase => {
                let mut specials: Vec<(String, Rank)> = self
                    .special_tokens()
                    .iter()
                    .map(|(s, r)| ((*s).to_string(), *r))
                    .collect();
                specials.extend((199998..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
                #[cfg(not(target_arch = "wasm32"))]
                {
                    load_encoding_from_file(
                        vocab_file_path,
                        check_hash.then(|| self.expected_hash()),
                        specials,
                        &self.pattern(),
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
                }
            }
            _ => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    load_encoding_from_file(
                        vocab_file_path,
                        check_hash.then(|| self.expected_hash()),
                        self.special_tokens().iter().cloned(),
                        &self.pattern(),
                    )
                }
                #[cfg(target_arch = "wasm32")]
                {
                    load_encoding_from_bytes(
                        &vocab_bytes,
                        None,
                        self.special_tokens().iter().cloned(),
                        &self.pattern(),
                    )
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn load(&self) -> Result<CoreBPE, LoadError> {
        let url = self.public_vocab_file_url();
        let vocab_bytes = download_or_find_cached_file_bytes(&url, Some(self.expected_hash()))
            .await
            .map_err(LoadError::DownloadOrLoadVocabFile)?;

        match self {
            Self::O200kHarmony => {
                let mut specials: Vec<(String, Rank)> = self
                    .special_tokens()
                    .iter()
                    .map(|(s, r)| ((*s).to_string(), *r))
                    .collect();
                specials.extend((200014..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
                load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
            }
            Self::O200kBase => {
                let mut specials: Vec<(String, Rank)> = self
                    .special_tokens()
                    .iter()
                    .map(|(s, r)| ((*s).to_string(), *r))
                    .collect();
                specials.extend((199998..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
                load_encoding_from_bytes(&vocab_bytes, None, specials, &self.pattern())
            }
            _ => load_encoding_from_bytes(
                &vocab_bytes,
                None,
                self.special_tokens().iter().cloned(),
                &self.pattern(),
            ),
        }
    }

    fn public_vocab_file_url(&self) -> String {
        let base = tiktoken_base_url();
        match self {
            Self::O200kBase => format!("{base}o200k_base.tiktoken"),
            Self::O200kHarmony => format!("{base}o200k_base.tiktoken"),
            Self::Cl100kBase => format!("{base}cl100k_base.tiktoken"),
        }
    }

    fn vocab_file_name(&self) -> &'static str {
        match self {
            Self::O200kBase => "o200k_base.tiktoken",
            Self::O200kHarmony => "o200k_base.tiktoken",
            Self::Cl100kBase => "cl100k_base.tiktoken",
        }
    }

    fn expected_hash(&self) -> &'static str {
        match self {
            Self::O200kBase => "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
            Self::O200kHarmony => {
                "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
            }
            Self::Cl100kBase => "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        }
    }

    fn special_tokens(&self) -> &'static [(&'static str, Rank)] {
        match self {
            Self::O200kBase => &[],
            Self::O200kHarmony => &[
                ("<|startoftext|>", 199998),
                ("<|endoftext|>", 199999),
                ("<|reserved_200000|>", 200000),
                ("<|reserved_200001|>", 200001),
                ("<|return|>", 200002),
                ("<|constrain|>", 200003),
                ("<|reserved_200004|>", 200004),
                ("<|channel|>", 200005),
                ("<|start|>", 200006),
                ("<|end|>", 200007),
                ("<|message|>", 200008),
                ("<|reserved_200009|>", 200009),
                ("<|reserved_200010|>", 200010),
                ("<|reserved_200011|>", 200011),
                ("<|call|>", 200012),
                ("<|reserved_200013|>", 200013),
            ],
            Self::Cl100kBase => &[
                ("<|endoftext|>", 100257),
                ("<|fim_prefix|>", 100258),
                ("<|fim_middle|>", 100259),
                ("<|fim_suffix|>", 100260),
                ("<|endofprompt|>", 100276),
            ],
        }
    }

    fn pattern(&self) -> String {
        match self {
            Self::O200kBase => {
                [
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+",
                ].join("|")
            }
            Self::O200kHarmony => {
                [
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+",
                ].join("|")
            }
            Self::Cl100kBase => {
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".to_string()
            }
        }
    }
}

fn load_tiktoken_vocab<R>(
    mut reader: R,
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error>
where
    R: std::io::BufRead,
{
    let mut hasher = expected_hash.map(|_| Sha256::new());
    let mut bpe_ranks = HashMap::new();
    // using readline here so that the line returned includes the newline bytes for the hasher
    let mut lin_no = 0;
    let mut line_buffer = String::new();
    while reader.read_line(&mut line_buffer)? > 0 {
        lin_no += 1;
        if let Some(hasher) = hasher.as_mut() {
            hasher.update(line_buffer.as_bytes());
        }
        let line = line_buffer.trim_end();
        let (token, rank) = line.split_once(' ').ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected token and rank, could not split on ' ' at line {lin_no}"),
            )
        })?;
        let bytes = BASE64_STANDARD.decode(token).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to decode base64 token at line {lin_no}: {e}",),
            )
        })?;
        let rank = rank.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to parse rank at line {lin_no}: {e}"),
            )
        })?;
        bpe_ranks.insert(bytes, rank);
        line_buffer.clear();
    }
    if let Some(hasher) = hasher {
        let expected_hash = expected_hash.unwrap();
        let computed_hash = format!("{:x}", hasher.finalize());
        if computed_hash != expected_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("hash mismatch: computed={computed_hash}, expected={expected_hash}"),
            ));
        }
    }
    Ok(bpe_ranks)
}

pub fn load_tiktoken_vocab_file<P>(
    path: P,
    expected_hash: Option<&str>,
) -> std::result::Result<HashMap<Vec<u8>, Rank>, std::io::Error>
where
    P: AsRef<Path>,
{
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    load_tiktoken_vocab(reader, expected_hash)
}

pub fn load_encoding_from_file<P, S, TS>(
    file_path: P,
    expected_hash: Option<&str>,
    special_tokens: S,
    pattern: &str,
) -> Result<CoreBPE, LoadError>
where
    P: AsRef<Path>,
    S: IntoIterator<Item = (TS, Rank)>,
    TS: Into<String>,
{
    let encoder = load_tiktoken_vocab_file(file_path, expected_hash)
        .map_err(LoadError::InvalidTiktokenVocabFile)?;
    CoreBPE::new(
        encoder,
        special_tokens.into_iter().map(|(k, v)| (k.into(), v)),
        pattern,
    )
    .map_err(LoadError::CoreBPECreationFailed)
}

/// This returns the path to a file containing the data at `url`. If the file is
/// cached, it is used. Otherwise, the file is downloaded and cached.
#[cfg(not(target_arch = "wasm32"))]
fn download_or_find_cached_file(
    url: &str,
    expected_hash: Option<&str>,
) -> Result<PathBuf, RemoteVocabFileError> {
    let cache_dir = resolve_cache_dir()?;
    let cache_path = resolve_cache_path(&cache_dir, url);
    let lock_path = cache_path.with_extension("lock");
    let lock_file = File::create(&lock_path).map_err(|e| {
        RemoteVocabFileError::IOError(format!("creating lock file {lock_path:?}"), e)
    })?;
    lock_file
        .lock_exclusive()
        .map_err(|e| RemoteVocabFileError::IOError(format!("locking file {lock_path:?}"), e))?;
    let result = (|| {
        if cache_path.exists() {
            if verify_file_hash(&cache_path, expected_hash)? {
                return Ok(cache_path);
            }
            let _ = std::fs::remove_file(&cache_path);
        }
        let hash = load_remote_file(url, &cache_path)?;
        if let Some(expected_hash) = expected_hash {
            if hash != expected_hash {
                let _ = std::fs::remove_file(&cache_path);
                return Err(RemoteVocabFileError::HashMismatch {
                    file_url: url.to_string(),
                    expected_hash: expected_hash.to_string(),
                    computed_hash: hash,
                });
            }
        }
        Ok(cache_path)
    })();
    let _ = fs2::FileExt::unlock(&lock_file);
    result
}

#[cfg(target_arch = "wasm32")]
async fn download_or_find_cached_file_bytes(
    url: &str,
    expected_hash: Option<&str>,
) -> Result<Vec<u8>, RemoteVocabFileError> {
    let bytes = load_remote_file_bytes(url).await?;
    if let Some(expected_hash) = expected_hash {
        let computed_hash = format!("{:x}", Sha256::digest(&bytes));
        if computed_hash != expected_hash {
            return Err(RemoteVocabFileError::HashMismatch {
                file_url: url.to_string(),
                expected_hash: expected_hash.to_string(),
                computed_hash,
            });
        }
    }
    Ok(bytes)
}

fn resolve_cache_dir() -> Result<PathBuf, RemoteVocabFileError> {
    // we use a different env var and a different default dir name to avoid
    // conflicts with the python tiktoken package, while sharing a cache dir
    // with the python tiktoken package is a desirable future goal, it is not
    // a priority and we should optimize for avoiding breaking tiktoken installs
    // on the same system until we can validate the correctness wrt the python
    // implementation and write tests to avoid regressions
    let cache_dir_override = std::env::var("TIKTOKEN_RS_CACHE_DIR").ok();
    if let Some(cache_dir_override) = cache_dir_override {
        Ok(PathBuf::from(cache_dir_override))
    } else {
        let cache_dir = std::env::temp_dir().join("tiktoken-rs-cache");
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            RemoteVocabFileError::IOError(format!("creating cache dir {cache_dir:?}"), e)
        })?;
        Ok(cache_dir)
    }
}

fn resolve_cache_path(cache_dir: &Path, url: &str) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(url.as_bytes());
    let cache_key = format!("{:x}", hasher.finalize());
    cache_dir.join(cache_key)
}

fn verify_file_hash(
    file_path: &Path,
    expected_hash: Option<&str>,
) -> Result<bool, RemoteVocabFileError> {
    let Some(expected_hash) = expected_hash else {
        return Ok(true);
    };
    let file = File::open(file_path)
        .map_err(|e| RemoteVocabFileError::IOError(format!("opening file {file_path:?}"), e))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    std::io::copy(&mut reader, &mut hasher).map_err(|e| {
        RemoteVocabFileError::IOError(format!("copying file {file_path:?} contents to hasher"), e)
    })?;
    let computed_hash = format!("{:x}", hasher.finalize());
    Ok(computed_hash == expected_hash)
}

/// Loads a remote file to `destination` and returns the computed hash of the
/// file contents.
#[cfg(not(target_arch = "wasm32"))]
fn load_remote_file(url: &str, destination: &Path) -> Result<String, RemoteVocabFileError> {
    let client = reqwest::blocking::Client::new();
    let mut response = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(e)))?;

    let file = File::create(destination)
        .map_err(|e| RemoteVocabFileError::IOError(format!("creating file {destination:?}"), e))?;
    let mut dest = BufWriter::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = response.read(&mut buffer).map_err(|e| {
            RemoteVocabFileError::IOError(format!("reading from response {url}"), e)
        })?;
        if bytes_read == 0 {
            break;
        }
        dest.write_all(&buffer[..bytes_read]).map_err(|e| {
            RemoteVocabFileError::IOError(format!("writing to file {destination:?}"), e)
        })?;
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(target_arch = "wasm32")]
fn load_remote_file(_url: &str, _destination: &Path) -> Result<String, RemoteVocabFileError> {
    Err(RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(
        Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Downloading files is not supported in wasm32",
        )),
    ))
}

#[cfg(target_arch = "wasm32")]
async fn load_remote_file_bytes(url: &str) -> Result<Vec<u8>, RemoteVocabFileError> {
    use reqwest::Client;

    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(e)))?;
    let bytes = response
        .bytes()
        .await
        .map_err(|e| RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(e)))?;
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_encodings() {
        for encoding in Encoding::all() {
            let _ = encoding.load().unwrap();
        }
    }

    #[test]
    fn test_parallel_load_encodings() {
        use std::thread;

        let encodings = Encoding::all();
        for encoding in encodings {
            let name = encoding.name();
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let name = name.to_string();
                    thread::spawn(move || {
                        Encoding::from_name(&name).unwrap().load().unwrap();
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("Thread panicked");
            }
        }
    }
}
