use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write as _},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{Duration, SystemTime},
};

use base64::{prelude::BASE64_STANDARD, Engine as _};

use crate::tiktoken::{CoreBPE, Rank};
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
    if cache_path.exists() {
        match verify_file_hash(&cache_path, expected_hash) {
            Ok(true) => return Ok(cache_path),
            // A partial file left by a writer that did not publish atomically.
            // The download below replaces it in one rename, so nothing removes
            // it first: removing it could take a peer's complete file instead.
            Ok(false) => {}
            // Gone between the check and the open: such a writer removed it.
            Err(RemoteVocabFileError::IOError(_, ref e))
                if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    load_remote_file(url, &cache_path, expected_hash)?;
    Ok(cache_path)
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

/// Downloads a remote file to `destination`, which another process may be
/// reading or writing at the same time. See `write_file_atomically`.
#[cfg(not(target_arch = "wasm32"))]
fn load_remote_file(
    url: &str,
    destination: &Path,
    expected_hash: Option<&str>,
) -> Result<(), RemoteVocabFileError> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| RemoteVocabFileError::FailedToDownloadOrLoadVocabFile(Box::new(e)))?;
    write_file_atomically(response, destination, url, expected_hash)
}

/// Streams `source` (read from `source_url`) into `destination` so that a
/// concurrent reader sees either no file or a complete, verified one.
///
/// The cache directory is shared by every process on the host, so several
/// processes that start with a cold cache download the same file at once.
/// Writing in place let one of them read another's partial download
/// ("invalid tiktoken vocab file: failed to parse rank"), or find the file
/// gone after a peer removed it as a hash mismatch. The bytes now go to a
/// temporary file unique to this process and call, in the destination's
/// directory, which is renamed over `destination` only after it is complete,
/// synced, and matches `expected_hash`. On any failure the temporary file is
/// removed and `destination` is left as it was.
///
/// The temporary file is created exclusively under a name no live writer
/// holds (`create_temp_file`), and a writer killed mid-download strands its
/// temporary file, so each download first sweeps the stale ones
/// (`remove_stale_temp_files`).
#[cfg(not(target_arch = "wasm32"))]
fn write_file_atomically(
    mut source: impl Read,
    destination: &Path,
    source_url: &str,
    expected_hash: Option<&str>,
) -> Result<(), RemoteVocabFileError> {
    remove_stale_temp_files(destination, SystemTime::now());
    let (file, temp_path) = create_temp_file(destination).map_err(|e| {
        RemoteVocabFileError::IOError(format!("creating a temporary file for {destination:?}"), e)
    })?;

    let mut write = |file: File| -> Result<(), RemoteVocabFileError> {
        let mut dest = BufWriter::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let bytes_read = source.read(&mut buffer).map_err(|e| {
                RemoteVocabFileError::IOError(format!("reading from response {source_url}"), e)
            })?;
            if bytes_read == 0 {
                break;
            }
            dest.write_all(&buffer[..bytes_read]).map_err(|e| {
                RemoteVocabFileError::IOError(format!("writing to file {temp_path:?}"), e)
            })?;
            hasher.update(&buffer[..bytes_read]);
        }
        let file = dest.into_inner().map_err(|e| {
            RemoteVocabFileError::IOError(format!("flushing file {temp_path:?}"), e.into_error())
        })?;
        file.sync_all()
            .map_err(|e| RemoteVocabFileError::IOError(format!("syncing file {temp_path:?}"), e))?;
        let computed_hash = format!("{:x}", hasher.finalize());
        if let Some(expected_hash) = expected_hash {
            if computed_hash != expected_hash {
                return Err(RemoteVocabFileError::HashMismatch {
                    file_url: source_url.to_string(),
                    expected_hash: expected_hash.to_string(),
                    computed_hash,
                });
            }
        }
        std::fs::rename(&temp_path, destination).map_err(|e| {
            RemoteVocabFileError::IOError(format!("renaming {temp_path:?} to {destination:?}"), e)
        })
    };
    let result = write(file);
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// How many temporary names a download tries before giving up.
#[cfg(not(target_arch = "wasm32"))]
const TEMP_FILE_ATTEMPTS: u32 = 16;

/// The temporary names a download of `destination` may use, in its
/// directory: `<name>.<pid>.<n>.tmp`. The process id keeps concurrent
/// processes apart and `n` keeps concurrent downloads in one process apart.
#[cfg(not(target_arch = "wasm32"))]
fn temp_file_candidates(destination: &Path) -> impl Iterator<Item = PathBuf> + '_ {
    let name = destination.file_name().map(|name| name.to_os_string());
    (0..TEMP_FILE_ATTEMPTS).filter_map(move |n| {
        let mut temp_name = name.clone()?;
        temp_name.push(format!(".{}.{n}.tmp", std::process::id()));
        Some(destination.with_file_name(temp_name))
    })
}

/// Creates the first free candidate name for a download of `destination`,
/// exclusively: `create_new` refuses any existing entry, a link included,
/// without following it. A taken name is skipped, never removed, because its
/// owner can be live: a process with the same id in another pid namespace,
/// or one on another host sharing the directory. Stranded names are left to
/// `remove_stale_temp_files`.
#[cfg(not(target_arch = "wasm32"))]
fn create_temp_file(destination: &Path) -> std::io::Result<(File, PathBuf)> {
    let mut taken = std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("no file name to derive a temporary name from in {destination:?}"),
    );
    for temp_path in temp_file_candidates(destination) {
        match File::options().write(true).create_new(true).open(&temp_path) {
            Ok(file) => return Ok((file, temp_path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => taken = e,
            Err(e) => return Err(e),
        }
    }
    Err(taken)
}

/// A temporary download untouched for this long was stranded by a writer
/// that died: a download writes continuously and completes in seconds.
#[cfg(not(target_arch = "wasm32"))]
const STALE_TEMP_FILE_AGE: Duration = Duration::from_secs(60 * 60);

/// Removes the temporary downloads of `destination` that no writer has
/// touched for `STALE_TEMP_FILE_AGE`, as of `now`. Nothing else sweeps the
/// shared cache directory, so each download sweeps before it writes.
#[cfg(not(target_arch = "wasm32"))]
fn remove_stale_temp_files(destination: &Path, now: SystemTime) {
    let (Some(dir), Some(name)) = (destination.parent(), destination.file_name()) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let prefix = format!("{}.", name.to_string_lossy());
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if !(file_name.starts_with(&prefix) && file_name.ends_with(".tmp")) {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= STALE_TEMP_FILE_AGE);
        if stale {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn load_remote_file(
    _url: &str,
    _destination: &Path,
    _expected_hash: Option<&str>,
) -> Result<(), RemoteVocabFileError> {
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
    use std::sync::atomic::Ordering;

    #[test]
    fn test_load_encodings() {
        for encoding in Encoding::all() {
            let _ = encoding.load().unwrap();
        }
    }
    /// A reader that yields `data` and then fails, like a dropped download.
    struct FailingReader<'a> {
        data: &'a [u8],
    }

    impl Read for FailingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.data.is_empty() {
                return Err(std::io::ErrorKind::ConnectionReset.into());
            }
            let n = self.data.len().min(buf.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            self.data = &self.data[n..];
            Ok(n)
        }
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "harmony-write-atomically-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn write_file_atomically_publishes_only_complete_verified_files() {
        let dir = scratch_dir("complete");
        let destination = dir.join("vocab");
        let body = b"rank data\n".repeat(4096);

        write_file_atomically(
            &body[..],
            &destination,
            "test://vocab",
            Some(&sha256_hex(&body)),
        )
        .unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), body);
        assert_eq!(entries(&dir), vec!["vocab".to_string()]);

        // A download that dies part way leaves the published file untouched
        // and no temporary file behind.
        let truncated = FailingReader {
            data: &body[..1000],
        };
        assert!(write_file_atomically(truncated, &destination, "test://vocab", None).is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), body);
        assert_eq!(entries(&dir), vec!["vocab".to_string()]);

        // So does one whose bytes do not match the expected hash.
        let err = write_file_atomically(
            &b"wrong"[..],
            &destination,
            "test://vocab",
            Some(&sha256_hex(&body)),
        )
        .unwrap_err();
        assert!(matches!(err, RemoteVocabFileError::HashMismatch { .. }));
        assert_eq!(std::fs::read(&destination).unwrap(), body);
        assert_eq!(entries(&dir), vec!["vocab".to_string()]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_file_atomically_never_exposes_a_partial_file() {
        let dir = scratch_dir("cold");
        let destination = dir.join("vocab");
        let body = b"rank data\n".repeat(4096);

        let truncated = FailingReader {
            data: &body[..1000],
        };
        assert!(write_file_atomically(truncated, &destination, "test://vocab", None).is_err());
        assert!(!destination.exists());
        assert!(entries(&dir).is_empty());

        // Concurrent writers of the same destination each publish a complete
        // file, and a reader polling the destination meanwhile only ever
        // finds no file or the whole of it. The barrier has the reader
        // polling before any writer starts, and the guard ends its polling
        // even when a writer panics, so a writer error fails the test rather
        // than leaving the scope waiting on the reader.
        struct StopOnDrop<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for StopOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        const WRITERS: usize = 4;
        let large = b"rank data\n".repeat(1 << 18);
        let writing = std::sync::atomic::AtomicBool::new(true);
        let start = std::sync::Barrier::new(WRITERS + 1);
        let complete_reads = std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                start.wait();
                let mut complete_reads = 0;
                while writing.load(Ordering::Acquire) {
                    match std::fs::read(&destination) {
                        Ok(read) => {
                            assert!(read == large, "read {} bytes", read.len());
                            complete_reads += 1;
                        }
                        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
                    }
                }
                complete_reads
            });
            let stop = StopOnDrop(&writing);
            let writers: Vec<_> = (0..WRITERS)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        for _ in 0..8 {
                            write_file_atomically(&large[..], &destination, "test://vocab", None)
                                .unwrap();
                        }
                    })
                })
                .collect();
            for writer in writers {
                writer.join().unwrap();
            }
            drop(stop);
            reader.join().unwrap()
        });
        assert!(complete_reads > 0, "the reader never saw a published file");
        assert_eq!(std::fs::read(&destination).unwrap(), large);
        assert_eq!(entries(&dir), vec!["vocab".to_string()]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn write_file_atomically_skips_a_taken_temp_name_without_removing_it() {
        let dir = scratch_dir("taken");
        let destination = dir.join("vocab");
        // Another writer (a process with this id in another pid namespace, or
        // a planted link) holds the first candidate name.
        let victim = dir.join("victim");
        std::fs::write(&victim, b"keep").unwrap();
        let taken = temp_file_candidates(&destination).next().unwrap();
        std::os::unix::fs::symlink(&victim, &taken).unwrap();

        let body = b"rank data\n".repeat(16);
        write_file_atomically(&body[..], &destination, "test://vocab", None).unwrap();
        assert_eq!(std::fs::read(&destination).unwrap(), body);
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep");
        assert!(std::fs::symlink_metadata(&taken).unwrap().file_type().is_symlink());
        let mut expected = vec![
            taken.file_name().unwrap().to_str().unwrap().to_string(),
            "victim".to_string(),
            "vocab".to_string(),
        ];
        expected.sort();
        assert_eq!(entries(&dir), expected);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_file_atomically_removes_stale_temp_files() {
        let dir = scratch_dir("stale");
        let destination = dir.join("vocab");
        let stale = dir.join("vocab.1.0.tmp");
        let live = dir.join("vocab.2.0.tmp");
        let other = dir.join("other.1.0.tmp");
        for path in [&stale, &live, &other] {
            std::fs::write(path, b"partial").unwrap();
        }
        let now = SystemTime::now();
        let age = |path: &Path, age: Duration| {
            File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(now - age)
                .unwrap();
        };
        age(&stale, STALE_TEMP_FILE_AGE);
        age(&live, STALE_TEMP_FILE_AGE / 2);

        remove_stale_temp_files(&destination, now);
        assert_eq!(entries(&dir), vec!["other.1.0.tmp", "vocab.2.0.tmp"]);

        // A download sweeps before it writes.
        age(&live, STALE_TEMP_FILE_AGE);
        let body = b"rank data\n".repeat(16);
        write_file_atomically(&body[..], &destination, "test://vocab", None).unwrap();
        assert_eq!(entries(&dir), vec!["other.1.0.tmp", "vocab"]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
