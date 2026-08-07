use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;

pub type Rank = u32;

/// Pieces at least this many bytes long use the heap-based merge, which avoids
/// the quadratic full-rescan/`Vec::remove` cost of the vector merge on long
/// runs (e.g. many repeated dashes). Shorter pieces keep the cache-friendly
/// vector path. Both paths produce identical token IDs.
const LARGE_PIECE_THRESHOLD: usize = 400;

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // This is a vector of (start, rank).
    // The rank is of the pair starting at position start.
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // n is often very small so considerations like cache-locality outweigh the algorithmic
    // complexity downsides of the `parts` vector.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    let parts = if piece.len() < LARGE_PIECE_THRESHOLD {
        _byte_pair_merge(ranks, piece)
    } else {
        _byte_pair_merge_heap(ranks, piece)
    };
    parts
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

/// Heap-based equivalent of [`_byte_pair_merge`] for large pieces.
///
/// It merges the leftmost lowest-rank pair at every step, exactly like the
/// vector implementation, but tracks boundaries with a doubly linked list and a
/// binary heap. This turns the per-merge full-vector rescan and `Vec::remove`
/// (O(mn) overall) into O(m log n), which removes the quadratic blow-up on long
/// runs such as sequences of dashes. The returned boundary vector is identical
/// to the one produced by `_byte_pair_merge`.
fn _byte_pair_merge_heap(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let n = piece.len();

    // Doubly linked list over current token start positions; `n` is the end
    // sentinel. `prev[0]` is `usize::MAX` (no predecessor).
    let mut next: Vec<usize> = (0..=n).map(|i| i + 1).collect();
    let mut prev: Vec<usize> = (0..=n).map(|i| i.wrapping_sub(1)).collect();
    let mut alive: Vec<bool> = vec![true; n];

    // Rank of merging the token starting at `p` with its successor, matching
    // `get_rank` in the vector implementation. `Rank::MAX` means "not mergeable".
    let pair_rank = |next: &[usize], p: usize| -> Rank {
        let q = next[p];
        if q >= n {
            return Rank::MAX;
        }
        let r = next[q];
        ranks.get(&piece[p..r]).copied().unwrap_or(Rank::MAX)
    };

    // Min-heap keyed by (rank, position): ties resolve to the leftmost pair, so
    // the merge order matches the vector scan's strict `<` (leftmost-min) rule.
    let mut heap: BinaryHeap<Reverse<(Rank, usize)>> = BinaryHeap::with_capacity(n);
    for p in 0..n.saturating_sub(1) {
        let rank = pair_rank(&next, p);
        if rank != Rank::MAX {
            heap.push(Reverse((rank, p)));
        }
    }

    while let Some(Reverse((rank, p))) = heap.pop() {
        if !alive[p] {
            continue;
        }
        // Lazy deletion: skip entries whose rank no longer matches the current
        // pair at `p`. The true current minimum always has a matching entry.
        if pair_rank(&next, p) != rank {
            continue;
        }

        let q = next[p];
        let r = next[q];
        alive[q] = false;
        next[p] = r;
        prev[r] = p;

        let rp = pair_rank(&next, p);
        if rp != Rank::MAX {
            heap.push(Reverse((rp, p)));
        }
        let pp = prev[p];
        if pp != usize::MAX {
            let rpp = pair_rank(&next, pp);
            if rpp != Rank::MAX {
                heap.push(Reverse((rpp, pp)));
            }
        }
    }

    let mut parts = Vec::with_capacity(n + 1);
    let mut p = 0;
    while p < n {
        parts.push((p, Rank::MAX));
        p = next[p];
    }
    parts.push((n, Rank::MAX));
    parts
}

// Various performance notes:
//
// Regex
// =====
// Most of the time is spent in regex. The easiest way to speed this up is by using less fancy
// regex features. For instance, using a regex parse-able by `regex` crate is 3x faster than
// the usual regex we use.
//
// However, given that we're using a regex parse-able by `regex`, there isn't much difference
// between using the `regex` crate and using the `fancy_regex` crate.
//
// There is an important interaction between threading, `regex` and `fancy_regex`.
// When using `fancy_regex`, we hit `regex.find_at`. It turns out that this causes contention on
// some mutable scratch space inside of `regex`. This absolutely kills performance. When using plain
// old `regex`, we don't hit this, because `find_iter` has a different code path.
// Related: https://github.com/rust-lang/regex/blob/master/PERFORMANCE.md
// Anyway, the way we get around this is with having a (mostly) thread local clone of the regex for
// each thread.
//
// Threading
// =========
// I tried using `rayon`. It wasn't really faster than using Python threads and releasing the GIL.
// So goodbye `rayon`! Let thread count etc be in control of our Python users.
//
// Caching
// =======
// The reference tokeniser has an lru cache over the equivalent of `byte_pair_encode`.
// Originally, we had one too! Without it, we were only vaguely faster than Python.
// I used an RWLock to protect the cache. This didn't seem to hurt single threaded performance
// noticeably, but it did affect multi-threaded performance. Weirdly, it seemed to affect
// multi-threaded performance even when I only had readers (maybed I messed something up?).
// Anyway, I realised that we could get rid of the cache, if we treat the set of tokens as a cache!
// These are exactly the set or merges that are likely to be hot. And now we don't have to think
// about interior mutability, memory use, or cloning.
//
// Hashing
// =======
// We use FxHashMap instead of the standard HashMap. This is maybe like a 5-10% win?
// The current implementation ends up doing a lot of hashing of bytes. In theory, this could be made
// to be hashing of two-tuples of ints, which looks like it may also be a couple percent faster.

struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread id counter
    // that works great for our use case of avoiding collisions in our array. Unfortunately,
    // it's private. However, there are only so many ways you can layout a u64, so just transmute
    // https://github.com/rust-lang/rust/issues/67939
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}
impl std::error::Error for DecodeKeyError {}

#[derive(Debug, Clone)]
pub struct DecodeError {
    pub message: String,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not decode tokens: {}", self.message)
    }
}
impl std::error::Error for DecodeError {}

const MAX_NUM_THREADS: usize = 128;

#[derive(Clone)]
pub struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    fn _get_tl_regex(&self) -> &Regex {
        // See performance notes above for what this is about
        // It's also a little janky, please make a better version of it!
        // However, it's nice that this doesn't leak memory to short-lived threads
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    pub fn decode_bytes<S, E>(&self, tokens: S) -> Result<Vec<u8>, DecodeKeyError>
    where
        S: IntoIterator<Item = E>,
        E: Borrow<Rank>,
    {
        let token_iter = tokens.into_iter();
        let (lower, _upper) = token_iter.size_hint();
        let mut ret = Vec::with_capacity(lower * 2);
        for token in token_iter {
            let &token = token.borrow();
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_tokens_decoder
                    .get(&token)
                    .ok_or(DecodeKeyError { token })?,
            };
            ret.extend(token_bytes);
        }
        Ok(ret)
    }

    pub fn decode_utf8<S, E>(&self, tokens: S) -> Result<String, DecodeError>
    where
        S: IntoIterator<Item = E>,
        E: Borrow<Rank>,
    {
        let bytes = self.decode_bytes(tokens).map_err(|e| DecodeError {
            message: format!("Invalid token error: {e}"),
        })?;
        String::from_utf8(bytes).map_err(|e| DecodeError {
            message: format!("Invalid utf-8 sequence: {e}"),
        })
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }

    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> (Vec<Rank>, usize) {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                // Find the next allowed special token, if any
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to encode_ordinary
            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        (ret, last_piece_token_len)
    }

    fn _increase_last_piece_token_len(
        &self,
        tokens: Vec<Rank>,
        mut last_piece_token_len: usize,
    ) -> (Vec<Rank>, usize) {
        // Unfortunately, the locations where our regex splits can be unstable.
        // For the purposes of determining unstable tokens, unstable regex splitting
        // is only a problem if a split that was present disappears, since this can
        // lead to merging of tokens otherwise thought to be stable.
        // cl100k_base makes our life hard by including the \s*[\r\n]+
        // pattern. This can e.g. cause "\n" + " " to become "\n \n".
        // Here is a quick and dirty fix:
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev()
                            .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len])
            {
                while (last_piece_token_len < tokens.len())
                    && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1])
                {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());

        (tokens, last_piece_token_len)
    }

    pub fn _encode_unstable_native(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (tokens, last_piece_token_len) = self.encode(text, allowed_special);
        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token and we have
            // no unstable bytes
            return (tokens, HashSet::new());
        }
        let (mut tokens, last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self
            .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
            .unwrap();
        tokens.truncate(tokens.len() - last_piece_token_len);

        // TODO: we should try harder to find additional stable tokens
        // This would reduce the amount of retokenising when determining completions
        // Refer to the logic in an older version of this file

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        // This is the easy bit. Just find all single tokens that start with unstable_bytes
        // (including tokens that exactly match unstable_bytes)
        // Separating this from the loop below helps with performance in a common case.
        let mut point = self
            .sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());
        while point < self.sorted_token_bytes.len()
            && self.sorted_token_bytes[point].starts_with(&unstable_bytes)
        {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()],
            ]);
            point += 1;
        }

        // Now apply even more brute force. At every (other) possible position for the straddling
        // token, concatenate additional bytes from that token (if any) to unstable_bytes,
        // and retokenise the whole thing and see what we get.
        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self
                .sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);
            // TODO: Perf optimisation if suffix starts with " "?
            while point < self.sorted_token_bytes.len()
                && self.sorted_token_bytes[point].starts_with(suffix)
            {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match std::str::from_utf8(&possibility) {
                    // Morally, this is byte_pair_encode(&possibility, &self.encoder)
                    // But we might have introduced a regex split which would prevent merges.
                    // (particularly possible in the presence of unstable regex splits)
                    // So convert to UTF-8 and do regex splitting.
                    // E.g. with cl100k_base "  !" gets split to " " + " !",
                    // but byte_pair_encode("  !") != byte_pair_encode(" ")
                    Ok(s) => self.encode_ordinary(s),

                    // Technically, whether or not this arm is correct depends on whether there
                    // would be a regex split before the UTF-8 truncation point.
                    // Probably niche enough that no one will ever notice (after all, people didn't
                    // notice all the big holes in the previous unstable token implementation)
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
                    // Something like the following is intriguing but incorrect:
                    // Err(e) => self.encode_ordinary(unsafe {
                    //     std::str::from_utf8_unchecked(&possibility[..e.valid_up_to()])
                    // }),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
            }
        }

        // This is also not straightforward. While we generally assume that regex splits are stable,
        // unfortunately, they are not. That is, if adding bytes were to make a split appear in
        // unstable_bytes, this could make tokens possible which our logic would otherwise think
        // would be merged.
        // For example, with gpt2, the use of \s+(?!\S) means that "\n\n" could
        // develop a split, e.g. "\n\n0" splits into "\n"+"\n"+"0", making "\n" a possible token.
        // Here is a quick and dirty fix:
        // This isn't right if we ever remove \s+(?!\S)
        if unstable_bytes.len() > 1 {
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0
                && last_decoded.0.is_some_and(|c| c.is_whitespace())
            {
                let mut reencoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                reencoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }

        (tokens, completions)
    }

    pub fn new<E, SE>(
        encoder: E,
        special_tokens_encoder: SE,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        E: IntoIterator<Item = (Vec<u8>, Rank)>,
        SE: IntoIterator<Item = (String, Rank)>,
    {
        Self::new_internal(
            HashMap::from_iter(encoder),
            HashMap::from_iter(special_tokens_encoder),
            pattern,
        )
    }

    fn new_internal(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let regex = Regex::new(pattern)?;

        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|"))?
        };

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert!(
            encoder.len() == decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        // Clone because I don't know how to tell Rust I'm not going to change the map
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect(),
            special_regex_tls: (0..MAX_NUM_THREADS)
                .map(|_| special_regex.clone())
                .collect(),
            sorted_token_bytes,
        })
    }

    pub fn special_tokens(&self) -> HashSet<&str> {
        self.special_tokens_encoder
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        let allowed_special = self.special_tokens();
        self.encode(text, &allowed_special).0
    }

    pub fn is_special_token(&self, token: Rank) -> bool {
        self.special_tokens_decoder.contains_key(&token)
    }
}

#[cfg(test)]
mod bpe_merge_tests {
    use super::{
        _byte_pair_merge, _byte_pair_merge_heap, byte_pair_encode, BinaryHeap, CoreBPE, Rank,
        Reverse, LARGE_PIECE_THRESHOLD,
    };
    use crate::{load_harmony_encoding, HarmonyEncodingName};
    use rustc_hash::FxHashMap as HashMap;
    use std::collections::HashSet;

    /// Boundary start positions produced by a merge function. `byte_pair_encode`
    /// only reads `part.0`, so equal start sequences mean identical tokens.
    fn starts(parts: &[(usize, Rank)]) -> Vec<usize> {
        parts.iter().map(|p| p.0).collect()
    }

    /// Simple deterministic xorshift so tests need no rand dependency.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
    }

    #[test]
    fn heap_matches_vector_on_synthetic_ranks() {
        // A small BPE table over a 4-symbol alphabet, including overlapping
        // pairs so tie-breaking (leftmost-lowest-rank) is exercised.
        let alphabet = [b'a', b'b', b'c', b'-'];
        let mut ranks: HashMap<Vec<u8>, Rank> = HashMap::default();
        let mut rank = 0u32;
        for &b in &alphabet {
            ranks.insert(vec![b], rank);
            rank += 1;
        }
        for &b0 in &alphabet {
            for &b1 in &alphabet {
                ranks.insert(vec![b0, b1], rank);
                rank += 1;
            }
        }
        for &b0 in &alphabet {
            for &b1 in &alphabet {
                for &b2 in &alphabet {
                    ranks.insert(vec![b0, b1, b2], rank);
                    rank += 1;
                }
            }
        }

        let mut rng = Rng(0x1234_5678_9abc_def0);
        for _ in 0..2000 {
            let len = 2 + (rng.next_u32() as usize % 64);
            let piece: Vec<u8> = (0..len)
                .map(|_| alphabet[rng.next_u32() as usize % alphabet.len()])
                .collect();
            let vec_parts = _byte_pair_merge(&ranks, &piece);
            let heap_parts = _byte_pair_merge_heap(&ranks, &piece);
            assert_eq!(
                starts(&vec_parts),
                starts(&heap_parts),
                "mismatch for piece {piece:?}"
            );
        }
    }

    #[test]
    fn heap_matches_vector_on_real_ranks() {
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let ranks = &encoding.tokenizer.encoder;

        // Long runs of a single byte are the pathological case from issue #79,
        // including lengths straddling the dispatch threshold.
        let mut lengths = vec![
            2usize,
            3,
            16,
            LARGE_PIECE_THRESHOLD - 1,
            LARGE_PIECE_THRESHOLD,
            LARGE_PIECE_THRESHOLD + 1,
            1024,
            4096,
        ];
        lengths.sort_unstable();
        for &len in &lengths {
            for &byte in &[b'-', b'a', b' ', b'\n'] {
                let piece = vec![byte; len];
                assert_eq!(
                    starts(&_byte_pair_merge(ranks, &piece)),
                    starts(&_byte_pair_merge_heap(ranks, &piece)),
                    "single-byte run mismatch: byte={byte} len={len}"
                );
            }
        }

        // Randomized printable-ASCII pieces across short and long lengths.
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for _ in 0..1500 {
            let len = 2 + (rng.next_u32() as usize % 900);
            let piece: Vec<u8> = (0..len)
                .map(|_| 0x20u8 + (rng.next_u32() as u8 % 0x5f))
                .collect();
            let vec_parts = _byte_pair_merge(ranks, &piece);
            let heap_parts = _byte_pair_merge_heap(ranks, &piece);
            assert_eq!(
                starts(&vec_parts),
                starts(&heap_parts),
                "random piece mismatch (len {len})"
            );
            // The public entry point must also agree with the vector merge.
            let via_entry = byte_pair_encode(&piece, ranks);
            let via_vector: Vec<Rank> = vec_parts
                .windows(2)
                .map(|w| ranks[&piece[w[0].0..w[1].0]])
                .collect();
            assert_eq!(
                via_entry, via_vector,
                "byte_pair_encode mismatch (len {len})"
            );
        }
    }

    #[test]
    fn heap_matches_vector_on_full_byte_range_random_pieces() {
        // Covers the entire 0..=255 byte space, including bytes that never
        // appear as valid standalone UTF-8 (e.g. lone continuation bytes).
        // `_byte_pair_merge`/`byte_pair_encode` operate on raw bytes and must
        // not panic or disagree regardless of what a regex piece contains.
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let ranks = &encoding.tokenizer.encoder;

        let mut rng = Rng(0x0ff5_e11e_b47e_5eed);
        for _ in 0..1500 {
            let len = 2 + (rng.next_u32() as usize % 900);
            let piece: Vec<u8> = (0..len).map(|_| rng.next_u32() as u8).collect();
            let vec_parts = _byte_pair_merge(ranks, &piece);
            let heap_parts = _byte_pair_merge_heap(ranks, &piece);
            assert_eq!(
                starts(&vec_parts),
                starts(&heap_parts),
                "full-byte-range piece mismatch (len {len})"
            );
        }
    }

    #[test]
    fn heap_matches_vector_on_multilingual_and_mixed_random_pieces() {
        // Correctness contract: cover repeated characters, whitespace,
        // punctuation, multilingual text, and mixed inputs.
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let ranks = &encoding.tokenizer.encoder;

        // Pool of byte chunks drawn from real multilingual/mixed text, so
        // random concatenations still contain realistic multi-byte UTF-8
        // structure, whitespace, punctuation, digits, and code-like tokens.
        let pool_text = concat!(
            "Hello, World! 123 456 789 --- ___ /// \\\\ ",
            "你好世界这是一个测试的中文文本 ",
            "مرحبا بالعالم هذا اختبار للنص العربي ",
            "Привет мир это тест русского текста ",
            "नमस्ते दुनिया यह हिन्दी पाठ परीक्षण है ",
            "こんにちは世界これは日本語のテストです ",
            "😀🎉👍🏽🚀❤️‍🔥🙂🙃😅 ",
            "Café naïve façade Zürich Straße résumé jalapeño ",
            "def foo(x, y):\n    return x + y  # code-ish\n",
        );
        let pool: Vec<u8> = pool_text.bytes().collect();

        let mut rng = Rng(0x0b5e_ed00_face_cafe);
        for _ in 0..1500 {
            let len = 2 + (rng.next_u32() as usize % 900);
            let piece: Vec<u8> = (0..len)
                .map(|_| pool[rng.next_u32() as usize % pool.len()])
                .collect();
            let vec_parts = _byte_pair_merge(ranks, &piece);
            let heap_parts = _byte_pair_merge_heap(ranks, &piece);
            assert_eq!(
                starts(&vec_parts),
                starts(&heap_parts),
                "multilingual/mixed piece mismatch (len {len})"
            );
        }

        // Long *contiguous* runs of a single non-dash multilingual unit -
        // the same pathology as issue #79 but for other scripts.
        let long_samples: Vec<Vec<u8>> = vec![
            "分".repeat(2000).into_bytes(),
            "🙂".repeat(1200).into_bytes(),
            "ك".repeat(2000).into_bytes(),
            "_".repeat(50_000).into_bytes(),
        ];
        for piece in &long_samples {
            assert_eq!(
                starts(&_byte_pair_merge(ranks, piece)),
                starts(&_byte_pair_merge_heap(ranks, piece)),
                "long multilingual run mismatch (len {})",
                piece.len()
            );
        }
    }

    /// Recomputes `encode_ordinary`'s token sequence for `text` while forcing
    /// the vector merge on every piece, independent of `byte_pair_encode`'s
    /// dispatch logic. This is a ground-truth oracle for the public API: if
    /// `CoreBPE::encode_ordinary`/`CoreBPE::encode` ever diverge from it, the
    /// dispatch (not just the low-level merge) has broken behavior.
    fn oracle_encode_ordinary_vector_only(bpe: &CoreBPE, text: &str) -> Vec<Rank> {
        let regex = bpe._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            match bpe.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => {
                    let parts = _byte_pair_merge(&bpe.encoder, piece);
                    ret.extend(
                        parts
                            .windows(2)
                            .map(|w| bpe.encoder[&piece[w[0].0..w[1].0]]),
                    );
                }
            }
        }
        ret
    }

    #[test]
    fn encode_and_encode_ordinary_match_vector_oracle_on_multilingual_and_mixed_text() {
        // Preserve `encode`/`encode_ordinary` behavior end-to-end (not just
        // the internal merge helper) on multilingual and mixed real text,
        // including runs that straddle `LARGE_PIECE_THRESHOLD`.
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let bpe = encoding.tokenizer();

        let mut samples: Vec<String> = vec![
            "你好，世界！这是一个测试。".repeat(5),
            "مرحبا بالعالم، هذا اختبار.".repeat(5),
            "Привет, мир! Это тест.".repeat(5),
            "नमस्ते दुनिया, यह एक परीक्षण है।".repeat(5),
            "こんにちは世界。これはテストです。".repeat(5),
            "😀🎉👍🏽🚀❤️‍🔥".repeat(80),
            "Café, naïve, façade, Zürich, Straße.".repeat(10),
            format!("Hello world 123 !!! {} more text 456.", "-".repeat(2000)),
            "分".repeat(1000),
            "🙂".repeat(600),
        ];
        // Lengths straddling LARGE_PIECE_THRESHOLD (400 bytes) for a 3-byte
        // CJK character: 133*3=399, 134*3=402.
        samples.push("あ".repeat(133));
        samples.push("あ".repeat(134));

        for text in &samples {
            let via_ordinary = bpe.encode_ordinary(text);
            let oracle = oracle_encode_ordinary_vector_only(bpe, text);
            assert_eq!(
                via_ordinary, oracle,
                "encode_ordinary mismatch for {text:?}"
            );

            let decoded = bpe.decode_utf8(via_ordinary.clone()).unwrap();
            assert_eq!(&decoded, text, "round-trip mismatch for {text:?}");

            // None of the samples contain special-token markers, so `encode`
            // with an empty allow-list must match `encode_ordinary` exactly.
            let (via_encode, _) = bpe.encode(text, &HashSet::new());
            assert_eq!(
                via_encode, via_ordinary,
                "encode/encode_ordinary mismatch for {text:?}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Complexity evidence: instrumented operation-count clones.
    //
    // A wall-clock sampling profiler (e.g. `samply`) was attempted for this
    // evidence but requires the Windows Performance Toolkit (`xperf`), which
    // needs an interactive elevated ADK install not available in this
    // environment. Counting the exact operations the code comment names -
    // "rescan" steps for the vector merge (O(mn)) and heap pops for the heap
    // merge (O(m log n)) - is a more precise, deterministic, and
    // machine-independent substitute for proving the asymptotic complexity
    // claim, and doubles as a regression guard against the old scaling
    // returning silently.
    // -----------------------------------------------------------------

    /// Verbatim copy of `_byte_pair_merge`'s algorithm, instrumented to count
    /// full-vector rescan steps (the `for (i, &(_, rank)) in ...` loop body
    /// that runs once per remaining part on every merge). This is exactly the
    /// operation the source comment calls out as making the function O(mn).
    fn _byte_pair_merge_counting(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> u64 {
        let mut rescan_steps: u64 = 0;
        let mut parts = Vec::with_capacity(piece.len() + 1);

        let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
        for i in 0..piece.len() - 1 {
            let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
            parts.push((i, rank));
        }
        parts.push((piece.len() - 1, Rank::MAX));
        parts.push((piece.len(), Rank::MAX));

        let get_rank = {
            #[inline(always)]
            |parts: &Vec<(usize, Rank)>, i: usize| {
                if (i + 3) < parts.len() {
                    *ranks
                        .get(&piece[parts[i].0..parts[i + 3].0])
                        .unwrap_or(&Rank::MAX)
                } else {
                    Rank::MAX
                }
            }
        };

        while min_rank.0 != Rank::MAX {
            let i = min_rank.1;
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1);
            }
            parts[i].1 = get_rank(&parts, i);
            parts.remove(i + 1);

            min_rank = (Rank::MAX, usize::MAX);
            for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                rescan_steps += 1;
                if rank < min_rank.0 {
                    min_rank = (rank, i);
                }
            }
        }
        rescan_steps
    }

    /// Verbatim copy of `_byte_pair_merge_heap`'s algorithm, instrumented to
    /// count `heap.pop()` calls - the operation that makes the function
    /// O(m log n) instead of O(mn).
    fn _byte_pair_merge_heap_counting(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> u64 {
        let mut heap_pops: u64 = 0;
        let n = piece.len();

        let mut next: Vec<usize> = (0..=n).map(|i| i + 1).collect();
        let mut prev: Vec<usize> = (0..=n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive: Vec<bool> = vec![true; n];

        let pair_rank = |next: &[usize], p: usize| -> Rank {
            let q = next[p];
            if q >= n {
                return Rank::MAX;
            }
            let r = next[q];
            ranks.get(&piece[p..r]).copied().unwrap_or(Rank::MAX)
        };

        let mut heap: BinaryHeap<Reverse<(Rank, usize)>> = BinaryHeap::with_capacity(n);
        for p in 0..n.saturating_sub(1) {
            let rank = pair_rank(&next, p);
            if rank != Rank::MAX {
                heap.push(Reverse((rank, p)));
            }
        }

        while let Some(Reverse((rank, p))) = heap.pop() {
            heap_pops += 1;
            if !alive[p] {
                continue;
            }
            if pair_rank(&next, p) != rank {
                continue;
            }

            let q = next[p];
            let r = next[q];
            alive[q] = false;
            next[p] = r;
            prev[r] = p;

            let rp = pair_rank(&next, p);
            if rp != Rank::MAX {
                heap.push(Reverse((rp, p)));
            }
            let pp = prev[p];
            if pp != usize::MAX {
                let rpp = pair_rank(&next, pp);
                if rpp != Rank::MAX {
                    heap.push(Reverse((rpp, pp)));
                }
            }
        }
        heap_pops
    }

    #[test]
    fn complexity_probe_vector_is_quadratic_heap_is_quasilinear_on_dash_runs() {
        // Sanity check first: the counting clones must be faithful copies of
        // the real algorithms, i.e. produce the same tokens.
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let ranks = &encoding.tokenizer.encoder;
        for &len in &[64usize, 1000, 5000] {
            let piece = vec![b'-'; len];
            assert_eq!(
                starts(&_byte_pair_merge(ranks, &piece)),
                starts(&_byte_pair_merge_heap(ranks, &piece)),
                "sanity: real implementations disagree at len {len}"
            );
        }

        let lengths = [512usize, 1024, 2048, 4096, 8192, 16384, 32768];
        eprintln!("\nissue #79 complexity evidence (dash runs, real o200k ranks):");
        eprintln!(
            "{:>8}  {:>14}  {:>12}  {:>10}  {:>10}",
            "dashes", "vector_rescans", "heap_pops", "v_ratio", "h_ratio"
        );

        let mut prev_vector: Option<u64> = None;
        let mut prev_heap: Option<u64> = None;
        let mut vector_ratios = vec![];
        let mut heap_ratios = vec![];
        for &len in &lengths {
            let piece = vec![b'-'; len];
            let vector_steps = _byte_pair_merge_counting(ranks, &piece);
            let heap_pops = _byte_pair_merge_heap_counting(ranks, &piece);

            let v_ratio = prev_vector.map(|p| vector_steps as f64 / p as f64);
            let h_ratio = prev_heap.map(|p| heap_pops as f64 / p as f64);
            eprintln!(
                "{:>8}  {:>14}  {:>12}  {:>10}  {:>10}",
                len,
                vector_steps,
                heap_pops,
                v_ratio
                    .map(|r| format!("{r:.2}x"))
                    .unwrap_or_else(|| "-".into()),
                h_ratio
                    .map(|r| format!("{r:.2}x"))
                    .unwrap_or_else(|| "-".into()),
            );
            if let Some(r) = v_ratio {
                vector_ratios.push(r);
            }
            if let Some(r) = h_ratio {
                heap_ratios.push(r);
            }
            prev_vector = Some(vector_steps);
            prev_heap = Some(heap_pops);
        }

        // Quadratic growth means each length doubling should multiply the
        // vector's rescan-step count by roughly 4x (asymptotically). Require
        // it clearly exceeds linear (2x) growth to confirm O(mn) behavior.
        let avg_vector_ratio = vector_ratios.iter().sum::<f64>() / vector_ratios.len() as f64;
        assert!(
            avg_vector_ratio > 3.0,
            "expected clearly super-linear (~4x) growth in vector rescan steps per doubling, got avg {avg_vector_ratio:.2}x"
        );

        // O(m log n) growth means each doubling should roughly double the
        // heap-pop count times a small constant, not quadruple it. Require it
        // stays well below the vector's quadratic growth.
        let avg_heap_ratio = heap_ratios.iter().sum::<f64>() / heap_ratios.len() as f64;
        assert!(
            avg_heap_ratio < 2.5,
            "expected near-linear (~2x) growth in heap-pop count per doubling, got avg {avg_heap_ratio:.2}x"
        );
    }
}
