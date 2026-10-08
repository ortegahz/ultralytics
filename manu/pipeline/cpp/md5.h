// Minimal MD5 (RFC 1321) used only as a bit-exactness fingerprint.
//
// The streaming hash must never be the reason a comparison fails, so `md5_self_test()`
// checks the two canonical RFC 1321 vectors at startup and the caller aborts on mismatch.
// It is validated against Python's hashlib, which is the other side of the comparison.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace gmcpp {

class MD5 {
public:
    MD5() { reset(); }

    void reset() {
        a_ = 0x67452301u;
        b_ = 0xefcdab89u;
        c_ = 0x98badcfeu;
        d_ = 0x10325476u;
        len_ = 0;
        buf_len_ = 0;
    }

    void update(const uint8_t* data, size_t n) {
        len_ += static_cast<uint64_t>(n);
        while (n > 0) {
            const size_t take = (64 - buf_len_ < n) ? (64 - buf_len_) : n;
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            n -= take;
            if (buf_len_ == 64) {
                transform(buf_);
                buf_len_ = 0;
            }
        }
    }

    void update(const std::string& s) { update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }

    std::string hex() {
        // Finalise on a copy so the object stays usable.
        MD5 tmp = *this;
        uint8_t pad[72];
        std::memset(pad, 0, sizeof(pad));
        pad[0] = 0x80;
        const uint64_t bits = tmp.len_ * 8;
        const size_t pad_len = (tmp.buf_len_ < 56) ? (56 - tmp.buf_len_) : (120 - tmp.buf_len_);
        tmp.update_raw(pad, pad_len);
        uint8_t tail[8];
        for (int i = 0; i < 8; ++i) tail[i] = static_cast<uint8_t>((bits >> (8 * i)) & 0xFF);
        tmp.update_raw(tail, 8);

        uint32_t words[4] = {tmp.a_, tmp.b_, tmp.c_, tmp.d_};
        static const char* kHex = "0123456789abcdef";
        std::string out;
        out.reserve(32);
        for (int w = 0; w < 4; ++w) {
            for (int i = 0; i < 4; ++i) {
                const uint8_t byte = static_cast<uint8_t>((words[w] >> (8 * i)) & 0xFF);
                out.push_back(kHex[byte >> 4]);
                out.push_back(kHex[byte & 0x0F]);
            }
        }
        return out;
    }

private:
    void update_raw(const uint8_t* data, size_t n) {
        while (n > 0) {
            const size_t take = (64 - buf_len_ < n) ? (64 - buf_len_) : n;
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            n -= take;
            if (buf_len_ == 64) {
                transform(buf_);
                buf_len_ = 0;
            }
        }
    }

    static uint32_t rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

    void transform(const uint8_t block[64]) {
        static const uint32_t K[64] = {
            0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au,
            0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
            0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u, 0xf61e2562u, 0xc040b340u,
            0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
            0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
            0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
            0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u, 0x289b7ec6u, 0xeaa127fau,
            0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
            0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u,
            0xffeff47du, 0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
            0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};
        static const int S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                                  5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                                  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                                  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
        uint32_t m[16];
        for (int i = 0; i < 16; ++i) {
            m[i] = static_cast<uint32_t>(block[4 * i]) | (static_cast<uint32_t>(block[4 * i + 1]) << 8) |
                   (static_cast<uint32_t>(block[4 * i + 2]) << 16) |
                   (static_cast<uint32_t>(block[4 * i + 3]) << 24);
        }
        uint32_t A = a_, B = b_, C = c_, D = d_;
        for (int i = 0; i < 64; ++i) {
            uint32_t F;
            int g;
            if (i < 16) {
                F = (B & C) | (~B & D);
                g = i;
            } else if (i < 32) {
                F = (D & B) | (~D & C);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                F = B ^ C ^ D;
                g = (3 * i + 5) % 16;
            } else {
                F = C ^ (B | ~D);
                g = (7 * i) % 16;
            }
            F = F + A + K[i] + m[g];
            A = D;
            D = C;
            C = B;
            B = B + rotl(F, S[i]);
        }
        a_ += A;
        b_ += B;
        c_ += C;
        d_ += D;
    }

    uint32_t a_, b_, c_, d_;
    uint64_t len_;
    uint8_t buf_[64];
    size_t buf_len_;
};

// RFC 1321 test vectors. A fingerprint bug must never masquerade as a feature mismatch.
inline bool md5_self_test(std::string* why = nullptr) {
    auto digest = [](const char* s) {
        MD5 h;
        h.update(reinterpret_cast<const uint8_t*>(s), std::strlen(s));
        return h.hex();
    };
    if (digest("") != "d41d8cd98f00b204e9800998ecf8427e") {
        if (why) *why = "md5(\"\") mismatch";
        return false;
    }
    if (digest("abc") != "900150983cd24fb0d6963f7d28e17f72") {
        if (why) *why = "md5(\"abc\") mismatch";
        return false;
    }
    // 80-char input crosses more than one 64-byte block.
    if (digest("12345678901234567890123456789012345678901234567890123456789012345678901234567890") !=
        "57edf4a22be3c955ac49da2e2107b67a") {
        if (why) *why = "md5(80-char) mismatch";
        return false;
    }
    return true;
}

}  // namespace gmcpp