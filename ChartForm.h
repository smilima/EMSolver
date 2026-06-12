//---------------------------------------------------------------------------
// ChartForm.h - line-chart window for S11 / input impedance plots
// (UI designed in ChartForm.dfm)
//---------------------------------------------------------------------------
#ifndef ChartFormH
#define ChartFormH

#include <System.Classes.hpp>
#include <Vcl.Forms.hpp>
#include <Vcl.Controls.hpp>
#include <Vcl.ExtCtrls.hpp>
#include <Vcl.StdCtrls.hpp>
#include <vector>

// One named curve: x values (GHz) and y values
struct ChartCurve
{
    String name;
    std::vector<double> x, y;
    TColor color;
};

class TChartForm : public TForm
{
__published:    // IDE-managed Components
    TPanel    *topBar;
    TComboBox *cbPage;
    TButton   *btnCsv;
    TPaintBox *paintBox;
    void __fastcall onPaint(TObject *Sender);
    void __fastcall onPageChange(TObject *Sender);
    void __fastcall onCsvClick(TObject *Sender);

public:
    __fastcall TChartForm(TComponent *Owner);

    // plot pages; each page has its own axis labels and curves
    struct Page
    {
        String title, yLabel;
        String xLabel;          // empty -> "Frequency (GHz)"
        std::vector<ChartCurve> curves;
    };
    std::vector<Page> pages;

    void refreshPages();    // call after filling 'pages'
};

#endif
