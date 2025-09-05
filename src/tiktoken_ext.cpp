#include "openai_harmony/tiktoken_ext.hpp"
#include "openai_harmony/registry.hpp"
#include <stdexcept>

namespace openai_harmony {

std::shared_ptr<CoreBPE> TiktokenExt::load_encoding(Encoding encoding) {
    switch (encoding) {
        case Encoding::O200kHarmony:
            return load_o200k_harmony();
        default:
            throw std::invalid_argument("Unknown encoding: " + std::to_string(static_cast<int>(encoding)));
    }
}

std::string TiktokenExt::get_encoding_name(Encoding encoding) {
    switch (encoding) {
        case Encoding::O200kHarmony:
            return "o200k_harmony";
        default:
            throw std::invalid_argument("Unknown encoding");
    }
}

std::shared_ptr<CoreBPE> TiktokenExt::load_o200k_harmony() {
    // Create basic encoder for o200k_harmony
    std::unordered_map<std::vector<uint8_t>, Rank, VectorHash> encoder;
    
    // Add basic ASCII characters (0-255)
    for (int i = 0; i < 256; ++i) {
        std::vector<uint8_t> byte = {static_cast<uint8_t>(i)};
        encoder[byte] = static_cast<Rank>(i);
    }
    
    // Add extended BPE vocabulary (simplified version)
    std::vector<std::pair<std::string, Rank>> bpe_merges = {
        // Common English bigrams
        {"th", 256}, {"he", 257}, {"in", 258}, {"er", 259}, {"an", 260},
        {"re", 261}, {"ed", 262}, {"nd", 263}, {"on", 264}, {"en", 265},
        {"at", 266}, {"ou", 267}, {"it", 268}, {"is", 269}, {"or", 270},
        {"ti", 271}, {"as", 272}, {"te", 273}, {"et", 274}, {"ng", 275},
        {"of", 276}, {"al", 277}, {"de", 278}, {"se", 279}, {"le", 280},
        {"to", 281}, {"ar", 282}, {"st", 283}, {"nt", 284}, {"ro", 285},
        {"ne", 286}, {"om", 287}, {"li", 288}, {"la", 289}, {"el", 290},
        {"ma", 291}, {"ri", 292}, {"ic", 293}, {"co", 294}, {"ca", 295},
        
        // Common trigrams
        {"the", 296}, {"and", 297}, {"ing", 298}, {"ion", 299}, {"tio", 300},
        {"ent", 301}, {"ati", 302}, {"for", 303}, {"her", 304}, {"ter", 305},
        {"hat", 306}, {"tha", 307}, {"ere", 308}, {"ate", 309}, {"his", 310},
        {"con", 311}, {"res", 312}, {"ver", 313}, {"all", 314}, {"ons", 315},
        {"nce", 316}, {"men", 317}, {"ive", 318}, {"ted", 319}, {"com", 320},
        
        // Common words
        {"that", 321}, {"with", 322}, {"have", 323}, {"this", 324}, {"will", 325},
        {"your", 326}, {"from", 327}, {"they", 328}, {"know", 329}, {"want", 330},
        {"been", 331}, {"good", 332}, {"much", 333}, {"some", 334}, {"time", 335},
        {"very", 336}, {"when", 337}, {"come", 338}, {"here", 339}, {"just", 340},
        {"like", 341}, {"long", 342}, {"make", 343}, {"many", 344}, {"over", 345},
        {"such", 346}, {"take", 347}, {"than", 348}, {"them", 349}, {"well", 350},
        {"were", 351}, {"what", 352}, {"year", 353}, {"work", 354}, {"world", 355},
        
        // Programming-related tokens
        {"def", 356}, {"class", 357}, {"import", 358}, {"from", 359}, {"return", 360},
        {"if", 361}, {"else", 362}, {"elif", 363}, {"for", 364}, {"while", 365},
        {"try", 366}, {"except", 367}, {"finally", 368}, {"with", 369}, {"as", 370},
        {"pass", 371}, {"break", 372}, {"continue", 373}, {"lambda", 374}, {"yield", 375},
        {"async", 376}, {"await", 377}, {"global", 378}, {"nonlocal", 379}, {"assert", 380},
        
        // JSON/markup tokens
        {"{", 381}, {"}", 382}, {"[", 383}, {"]", 384}, {":", 385},
        {",", 386}, {"\"", 387}, {"'", 388}, {"<", 389}, {">", 390},
        {"</", 391}, {"/>", 392}, {"<!--", 393}, {"-->", 394}, {"&", 395},
        
        // Mathematical symbols
        {"+", 396}, {"-", 397}, {"*", 398}, {"/", 399}, {"=", 400},
        {"==", 401}, {"!=", 402}, {"<=", 403}, {">=", 404}, {"&&", 405},
        {"||", 406}, {"++", 407}, {"--", 408}, {"+=", 409}, {"-=", 410},
        
        // Whitespace and newlines
        {"  ", 411}, {"   ", 412}, {"    ", 413}, {"\n", 414}, {"\n\n", 415},
        {"\t", 416}, {"\r\n", 417}, {" \n", 418}, {"\n ", 419}, {"  \n", 420}
    };
    
    // Add BPE merges to encoder
    for (const auto& [text, rank] : bpe_merges) {
        std::vector<uint8_t> bytes(text.begin(), text.end());
        encoder[bytes] = rank;
    }
    
    // Create special tokens for harmony format
    std::unordered_map<std::string, Rank> special_tokens = {
        {"<|start|>", 200000},
        {"<|end|>", 200001},
        {"<|message|>", 200002},
        {"<|channel|>", 200003},
        {"<|constrain|>", 200004},
        {"<|call|>", 200005},
        {"<|reasoning|>", 200006},
        {"<|thinking|>", 200007},
        {"<|analysis|>", 200008},
        {"<|commentary|>", 200009},
        {"<|final|>", 200010},
        {"<|system|>", 200011},
        {"<|user|>", 200012},
        {"<|assistant|>", 200013},
        {"<|developer|>", 200014},
        {"<|tool|>", 200015},
        {"<|im_start|>", 200016},
        {"<|im_end|>", 200017},
        {"<|endoftext|>", 200018},
        {"<|startoftext|>", 200019}
    };
    
    // Create regex pattern for tokenization (GPT-4 style)
    std::string pattern = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)";
    
    return std::make_shared<CoreBPE>(encoder, special_tokens, pattern);
}

} // namespace openai_harmony
