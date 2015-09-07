#include "easybmp/EasyBMP.h"

#include "picture.h"
#include "fileIO.h"

    void TPicture::Write(TFileWriter& fw) const
    {
        DrawLine(fw.GetHandle());
    }

    void TPicture::SaveBMP(const std::string& s) const
    {
        BMP bmp;
        bmp.SetSize(SIZE, SIZE);
        for (size_t i = 0; i < SIZE; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                RGBApixel pixel;
                pixel.Blue = 255 - Get(i, j);
                pixel.Red = 255 - Get(i, j);
                pixel.Green = 255 - Get(i, j);
                pixel.Alpha = 0;
                bmp.SetPixel(j, i, pixel);
            }
        }
        if (!bmp.WriteToFile(s.c_str()))
        {
            throw TException("Could not SaveBMP '" + s + "'");
        }
    }
