//---------------------------------------------------------------------------

#include <vcl.h>
#pragma hdrstop
#include <tchar.h>
#include "SelfTest.h"
//---------------------------------------------------------------------------
// On hybrid-graphics laptops (Intel iGPU + NVIDIA/AMD dGPU) OpenGL contexts
// default to the integrated GPU. Exporting these symbols tells the NVIDIA
// Optimus / AMD PowerXpress drivers to run this process on the discrete GPU.
extern "C"
{
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int   AmdPowerXpressRequestHighPerformance = 1;
}
//---------------------------------------------------------------------------
USEFORM("MainUnit.cpp", MainForm);
//---------------------------------------------------------------------------
int WINAPI _tWinMain(HINSTANCE, HINSTANCE, LPTSTR, int)
{
    try
    {
        if (ParamCount() >= 1 && ParamStr(1).CompareIC(L"-selftest") == 0)
            return RunSelfTest();
        Application->Initialize();
        Application->MainFormOnTaskBar = true;
        Application->CreateForm(__classid(TMainForm), &MainForm);
        Application->Run();
    }
    catch (Exception &exception)
    {
        Application->ShowException(&exception);
    }
    catch (...)
    {
        try
        {
            throw Exception("");
        }
        catch (Exception &exception)
        {
            Application->ShowException(&exception);
        }
    }
    return 0;
}
//---------------------------------------------------------------------------
