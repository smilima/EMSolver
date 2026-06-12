//---------------------------------------------------------------------------
// ChartForm.cpp - line chart window (axes, grid, multi-curve, CSV export)
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "ChartForm.h"
#include <Vcl.Dialogs.hpp>
#include <cmath>
#include <cstdio>
#include <algorithm>

#pragma package(smart_init)
#pragma resource "*.dfm"

__fastcall TChartForm::TChartForm(TComponent *Owner)
    : TForm(Owner)
{
}

//---------------------------------------------------------------------------
void TChartForm::refreshPages()
{
    int keep = cbPage->ItemIndex;
    cbPage->Items->Clear();
    for (const auto &p : pages)
        cbPage->Items->Add(p.title);
    if (!pages.empty())
        cbPage->ItemIndex = (keep >= 0 && keep < (int)pages.size()) ? keep : 0;
    paintBox->Invalidate();
}

//---------------------------------------------------------------------------
void __fastcall TChartForm::onPageChange(TObject *)
{
    paintBox->Invalidate();
}

//---------------------------------------------------------------------------
static String FmtTick(double v)
{
    if (std::fabs(v) >= 1000 || (std::fabs(v) < 0.01 && v != 0))
        return String().sprintf(L"%.3g", v);
    return String().sprintf(L"%.4g", v);
}

//---------------------------------------------------------------------------
void __fastcall TChartForm::onPaint(TObject *)
{
    TCanvas *cv = paintBox->Canvas;
    int W = paintBox->Width, H = paintBox->Height;
    cv->Brush->Color = clWhite;
    cv->FillRect(TRect(0, 0, W, H));
    int pi = cbPage->ItemIndex;
    if (pi < 0 || pi >= (int)pages.size())
        return;
    const Page &pg = pages[pi];

    // data ranges
    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
    for (const auto &c : pg.curves)
        for (size_t i = 0; i < c.x.size(); ++i)
        {
            if (!std::isfinite(c.x[i]) || !std::isfinite(c.y[i]))
                continue;
            x0 = std::min(x0, c.x[i]); x1 = std::max(x1, c.x[i]);
            y0 = std::min(y0, c.y[i]); y1 = std::max(y1, c.y[i]);
        }
    if (x0 > x1 || y0 > y1)
        return;
    if (y1 - y0 < 1e-12) { y0 -= 1; y1 += 1; }
    double yPad = 0.06 * (y1 - y0);
    y0 -= yPad; y1 += yPad;

    int dpi = Screen->PixelsPerInch;
    int mL = MulDiv(72, dpi, 96), mR = MulDiv(20, dpi, 96);
    int mT = MulDiv(16, dpi, 96), mB = MulDiv(48, dpi, 96);
    int pw = W - mL - mR, phh = H - mT - mB;
    if (pw < 50 || phh < 50)
        return;

    auto X = [&](double x) { return mL + (int)((x - x0) / (x1 - x0) * pw); };
    auto Y = [&](double y) { return mT + phh - (int)((y - y0) / (y1 - y0) * phh); };

    // grid + ticks
    cv->Font->Color = clBlack;
    cv->Font->Size = 8;
    cv->Pen->Color = (TColor)0x00E0E0E0;
    cv->Pen->Style = psSolid;
    const int nTicks = 8;
    for (int t = 0; t <= nTicks; ++t)
    {
        double xv = x0 + (x1 - x0) * t / nTicks;
        int px = X(xv);
        cv->MoveTo(px, mT);
        cv->LineTo(px, mT + phh);
        String s = FmtTick(xv);
        cv->Brush->Style = bsClear;
        cv->TextOutW(px - cv->TextWidth(s) / 2, mT + phh + 6, s);
    }
    for (int t = 0; t <= 6; ++t)
    {
        double yv = y0 + (y1 - y0) * t / 6;
        int py = Y(yv);
        cv->Pen->Color = (TColor)0x00E0E0E0;
        cv->MoveTo(mL, py);
        cv->LineTo(mL + pw, py);
        String s = FmtTick(yv);
        cv->Brush->Style = bsClear;
        cv->TextOutW(mL - cv->TextWidth(s) - 6, py - cv->TextHeight(s) / 2, s);
    }
    // axes box
    cv->Pen->Color = clGray;
    cv->MoveTo(mL, mT); cv->LineTo(mL, mT + phh);
    cv->LineTo(mL + pw, mT + phh); cv->LineTo(mL + pw, mT);
    cv->LineTo(mL, mT);

    // labels
    cv->Font->Size = 9;
    cv->Brush->Style = bsClear;
    String xl = pg.xLabel.IsEmpty() ? String(L"Frequency (GHz)") : pg.xLabel;
    cv->TextOutW(mL + pw / 2 - cv->TextWidth(xl) / 2,
                 H - cv->TextHeight(xl) - 4, xl);
    cv->TextOutW(mL, 2, pg.yLabel);

    // curves
    for (const auto &c : pg.curves)
    {
        cv->Pen->Color = c.color;
        cv->Pen->Width = 2;
        bool first = true;
        for (size_t i = 0; i < c.x.size(); ++i)
        {
            if (!std::isfinite(c.x[i]) || !std::isfinite(c.y[i]))
            {
                first = true;
                continue;
            }
            int px = X(c.x[i]), py = Y(c.y[i]);
            if (first) { cv->MoveTo(px, py); first = false; }
            else         cv->LineTo(px, py);
        }
        cv->Pen->Width = 1;
    }
    // legend
    int lx = mL + 12, ly = mT + 8;
    for (const auto &c : pg.curves)
    {
        cv->Pen->Color = c.color;
        cv->Pen->Width = 3;
        cv->MoveTo(lx, ly + 7);
        cv->LineTo(lx + 24, ly + 7);
        cv->Pen->Width = 1;
        cv->Brush->Style = bsClear;
        cv->TextOutW(lx + 30, ly, c.name);
        ly += cv->TextHeight(c.name) + 4;
    }
}

//---------------------------------------------------------------------------
void __fastcall TChartForm::onCsvClick(TObject *)
{
    int pi = cbPage->ItemIndex;
    if (pi < 0 || pi >= (int)pages.size())
        return;
    const Page &pg = pages[pi];
    if (pg.curves.empty())
        return;
    std::unique_ptr<TSaveDialog> dlg(new TSaveDialog(this));
    dlg->Filter = L"CSV files (*.csv)|*.csv";
    dlg->DefaultExt = L"csv";
    dlg->FileName = L"plot.csv";
    if (!dlg->Execute())
        return;
    AnsiString path(dlg->FileName);
    FILE *f = fopen(path.c_str(), "w");
    if (!f)
        return;
    String xl = pg.xLabel.IsEmpty() ? String(L"freq_GHz") : pg.xLabel;
    xl = StringReplace(xl, L" ", L"_", TReplaceFlags() << rfReplaceAll);
    xl = StringReplace(xl, L",", L"_", TReplaceFlags() << rfReplaceAll);
    fprintf(f, "%s", AnsiString(xl).c_str());
    for (const auto &c : pg.curves)
    {
        AnsiString n(c.name);
        fprintf(f, ",%s", n.c_str());
    }
    fprintf(f, "\n");
    size_t rows = 0;
    for (const auto &c : pg.curves)
        rows = std::max(rows, c.x.size());
    for (size_t r = 0; r < rows; ++r)
    {
        if (!pg.curves[0].x.empty() && r < pg.curves[0].x.size())
            fprintf(f, "%.9g", pg.curves[0].x[r]);
        for (const auto &c : pg.curves)
            fprintf(f, ",%s", r < c.y.size()
                    ? AnsiString().sprintf("%.9g", c.y[r]).c_str() : "");
        fprintf(f, "\n");
    }
    fclose(f);
}
