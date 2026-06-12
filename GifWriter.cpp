//---------------------------------------------------------------------------
// GifWriter.cpp - GIF89a animation encoder
//
// Global 256-color palette: 6x7x6 RGB cube (252 used). LZW per GIF spec
// with 8-bit minimum code size. Frames are written as they arrive, so
// memory use is one frame regardless of recording length.
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GifWriter.h"
#include <cstring>
#include <algorithm>

#pragma package(smart_init)

static const int RL = 6, GL_ = 7, BL = 6;     // palette levels per channel

static inline uint8_t quantIdx(int r, int g, int b)
{
    int ri = (r * (RL - 1) + 127) / 255;
    int gi = (g * (GL_ - 1) + 127) / 255;
    int bi = (b * (BL - 1) + 127) / 255;
    return (uint8_t)(ri * (GL_ * BL) + gi * BL + bi);
}

//---------------------------------------------------------------------------
bool GifWriter::begin(const std::wstring &path, int width, int height,
                      int delayCs)
{
    finish();
    f = _wfopen(path.c_str(), L"wb");
    if (!f)
        return false;
    gw = width;
    gh = height;
    delay = delayCs;
    frames = 0;

    fwrite("GIF89a", 1, 6, f);
    putWord((uint16_t)gw);
    putWord((uint16_t)gh);
    putByte(0xF7);              // GCT present, 8 bits, 256 entries
    putByte(0);                 // background
    putByte(0);                 // aspect

    // global color table
    for (int i = 0; i < 256; ++i)
    {
        if (i < RL * GL_ * BL)
        {
            int ri = i / (GL_ * BL), gi = (i / BL) % GL_, bi = i % BL;
            putByte((uint8_t)(ri * 255 / (RL - 1)));
            putByte((uint8_t)(gi * 255 / (GL_ - 1)));
            putByte((uint8_t)(bi * 255 / (BL - 1)));
        }
        else
        {
            putByte(0); putByte(0); putByte(0);
        }
    }

    // NETSCAPE looping extension (loop forever)
    putByte(0x21); putByte(0xFF); putByte(11);
    fwrite("NETSCAPE2.0", 1, 11, f);
    putByte(3); putByte(1); putWord(0); putByte(0);
    return true;
}

//---------------------------------------------------------------------------
bool GifWriter::addFrame(const unsigned char *rgb, int width, int height,
                         int shrink)
{
    if (!f)
        return false;
    if (shrink < 1)
        shrink = 1;
    int w = width / shrink, h = height / shrink;
    if (w != gw || h != gh)
    {
        // frame size changed (window resized) - skip frame
        if (frames == 0) { gw = w; gh = h; }
        else return false;
    }

    // downscale (box average) + quantize
    std::vector<uint8_t> idx((size_t)gw * gh);
    for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x)
        {
            int r = 0, gg = 0, b = 0, n = 0;
            for (int dy = 0; dy < shrink; ++dy)
            {
                int sy = y * shrink + dy;
                if (sy >= height) break;
                const unsigned char *row = rgb + ((size_t)sy * width) * 3;
                for (int dx = 0; dx < shrink; ++dx)
                {
                    int sx = x * shrink + dx;
                    if (sx >= width) break;
                    r += row[sx * 3];
                    gg += row[sx * 3 + 1];
                    b += row[sx * 3 + 2];
                    ++n;
                }
            }
            if (n == 0) n = 1;
            idx[(size_t)y * gw + x] = quantIdx(r / n, gg / n, b / n);
        }

    // graphic control extension
    putByte(0x21); putByte(0xF9); putByte(4);
    putByte(0x04);              // no transparency, do not dispose
    putWord((uint16_t)delay);
    putByte(0); putByte(0);

    // image descriptor
    putByte(0x2C);
    putWord(0); putWord(0);
    putWord((uint16_t)gw); putWord((uint16_t)gh);
    putByte(0);                 // no local color table

    writeLzwImage(idx);
    ++frames;
    return true;
}

//---------------------------------------------------------------------------
void GifWriter::writeLzwImage(const std::vector<uint8_t> &indices)
{
    const int minCode = 8;
    const int clearCode = 1 << minCode;        // 256
    const int endCode = clearCode + 1;         // 257
    putByte((uint8_t)minCode);

    // child[code][symbol] -> next code, flat table
    static std::vector<int16_t> child;
    child.assign(4096 * 256, -1);

    uint8_t block[256];
    int blockLen = 0;
    uint32_t bitBuf = 0;
    int bitCnt = 0;

    auto flushBlock = [&]()
    {
        if (blockLen > 0)
        {
            putByte((uint8_t)blockLen);
            fwrite(block, 1, blockLen, f);
            blockLen = 0;
        }
    };
    auto putBits = [&](int code, int width)
    {
        bitBuf |= (uint32_t)code << bitCnt;
        bitCnt += width;
        while (bitCnt >= 8)
        {
            block[blockLen++] = (uint8_t)(bitBuf & 0xFF);
            bitBuf >>= 8;
            bitCnt -= 8;
            if (blockLen == 255)
                flushBlock();
        }
    };

    int codeWidth = minCode + 1;
    int nextCode = endCode + 1;
    putBits(clearCode, codeWidth);

    int cur = indices.empty() ? 0 : indices[0];
    for (size_t i = 1; i < indices.size(); ++i)
    {
        int sym = indices[i];
        int nxt = child[(size_t)cur * 256 + sym];
        if (nxt >= 0)
        {
            cur = nxt;
            continue;
        }
        putBits(cur, codeWidth);
        if (nextCode < 4096)
        {
            child[(size_t)cur * 256 + sym] = (int16_t)nextCode;
            if (nextCode == (1 << codeWidth) && codeWidth < 12)
                ++codeWidth;
            ++nextCode;
        }
        else
        {
            putBits(clearCode, codeWidth);
            std::fill(child.begin(), child.end(), (int16_t)-1);
            codeWidth = minCode + 1;
            nextCode = endCode + 1;
        }
        cur = sym;
    }
    putBits(cur, codeWidth);
    putBits(endCode, codeWidth);
    if (bitCnt > 0)
    {
        block[blockLen++] = (uint8_t)(bitBuf & 0xFF);
        if (blockLen == 255)
            flushBlock();
    }
    flushBlock();
    putByte(0);                 // block terminator
}

//---------------------------------------------------------------------------
void GifWriter::finish()
{
    if (!f)
        return;
    putByte(0x3B);              // trailer
    fclose(f);
    f = nullptr;
}
