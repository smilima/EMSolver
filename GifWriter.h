//---------------------------------------------------------------------------
// GifWriter.h - streaming animated GIF89a encoder (fixed 6x7x6 color cube)
//---------------------------------------------------------------------------
#ifndef GifWriterH
#define GifWriterH

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

class GifWriter
{
public:
    ~GifWriter() { finish(); }

    // delayCs = frame delay in 1/100 s
    bool begin(const std::wstring &path, int width, int height, int delayCs);
    // rgb: width*height*3, top-down rows. Downscaled by 'shrink' (>=1).
    bool addFrame(const unsigned char *rgb, int width, int height, int shrink);
    void finish();

    bool isOpen()     const { return f != nullptr; }
    int  frameCount() const { return frames; }

private:
    void writeLzwImage(const std::vector<uint8_t> &indices);
    void putByte(uint8_t b) { fputc(b, f); }
    void putWord(uint16_t w) { fputc(w & 0xFF, f); fputc(w >> 8, f); }

    FILE *f = nullptr;
    int   gw = 0, gh = 0, delay = 10;
    int   frames = 0;
};

#endif
