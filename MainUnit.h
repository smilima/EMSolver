//---------------------------------------------------------------------------
// MainUnit.h - RF Simulator main window (UI designed in MainUnit.dfm)
//---------------------------------------------------------------------------
#ifndef MainUnitH
#define MainUnitH
//---------------------------------------------------------------------------
#include <System.Classes.hpp>
#include <Vcl.Controls.hpp>
#include <Vcl.StdCtrls.hpp>
#include <Vcl.Forms.hpp>
#include <Vcl.ExtCtrls.hpp>
#include <Vcl.ComCtrls.hpp>
#include <Vcl.ValEdit.hpp>
#include <Vcl.Dialogs.hpp>
#include <Vcl.Grids.hpp>
#include <Vcl.Menus.hpp>
#include <Vcl.ToolWin.hpp>
#include <memory>
#include <thread>
#include <vector>

#include "Geometry.h"
#include "AntennaLib.h"
#include "TlmSolver.h"
#include "FdtdSolver.h"
#include "FemSolver.h"
#include "MomSolver.h"
#include "MomSurface.h"
#include "GLView.h"
#include "GifWriter.h"

class TChartForm;

//---------------------------------------------------------------------------
class TMainForm : public TForm
{
__published:    // IDE-managed Components
    TPanel        *topPanel;
    TLabel        *lblComponent;
    TLabel        *lblExcitation;
    TLabel        *lblSolver;
    TLabel        *lblDisplay;
    TComboBox     *cbAddKind;
    TToolButton   *btnAdd;
    TToolButton   *btnImport;
    TToolButton   *btnDelete;
    TComboBox     *cbExcitation;
    TComboBox     *cbWaveform;
    TToolButton   *btnRun;
    TToolButton   *btnStop;
    TComboBox     *cbCurrents;
    TCheckBox     *chkModel;
    TCheckBox     *chkDb;
    TCheckBox     *chkPlane;
    TToolButton   *btnZoom;
    TToolButton   *btnPlots;
    TToolButton   *btnFarField;
    TToolButton   *btnRecGif;
    TCheckBox     *chkGpu;
    TComboBox     *cbDbRange;
    TComboBox     *cbSolver;
    TCheckBox     *chkMesh;
    TToolButton   *btnMesh;
    TToolButton   *ToolSep1;
    TToolButton   *ToolSep2;
    TToolButton   *ToolSep3;
    TToolButton   *ToolSep4;
    TPanel        *leftPanel;
    TLabel        *lblComponents;
    TLabel        *lblProps;
    TLabel        *lblSim;
    TListBox      *lbObjects;
    TValueListEditor *vleProps;
    TButton       *btnApply;
    TValueListEditor *vleSim;
    TSplitter     *splitter;
    TPanel        *pnlView;
    TPanel        *playPanel;
    TLabel        *lblPlayback;
    TLabel        *lblFrame;
    TButton       *btnPlay;
    TTrackBar     *tbPlayback;
    TStatusBar    *statusBar;
    TTimer        *uiTimer;
	TMainMenu *MainMenu1;
	TMenuItem *File1;
	TMenuItem *File2;
	TMenuItem *Close1;
	TMenuItem *Close2;
	TMenuItem *SaveAs1;
	TMenuItem *SaveAs2;
	TMenuItem *N1;
	TMenuItem *N2;
	TToolBar *ToolBar1;
	TPopupMenu *pmObjects;
	TMenuItem *miDelete;
    void __fastcall OnAddClick(TObject *Sender);
    void __fastcall OnImportClick(TObject *Sender);
    void __fastcall OnDeleteClick(TObject *Sender);
    void __fastcall OnObjectsKeyDown(TObject *Sender, System::Word &Key,
                                     TShiftState Shift);
    void __fastcall OnObjectDelete(TObject *Sender);
    void __fastcall OnRunClick(TObject *Sender);
    void __fastcall OnStopClick(TObject *Sender);
    void __fastcall OnZoomClick(TObject *Sender);
    void __fastcall OnApplyClick(TObject *Sender);
    void __fastcall OnObjectSelect(TObject *Sender);
    void __fastcall OnSolverChanged(TObject *Sender);
    void __fastcall OnViewOptionChanged(TObject *Sender);
    void __fastcall OnPlotsClick(TObject *Sender);
    void __fastcall OnFarFieldClick(TObject *Sender);
    void __fastcall OnRecGifClick(TObject *Sender);
    void __fastcall OnPlaybackChange(TObject *Sender);
    void __fastcall OnPlayClick(TObject *Sender);
    void __fastcall OnTimerTick(TObject *Sender);
    void __fastcall OnFormClose(TObject *Sender, TCloseAction &Action);
    void __fastcall OnExcitationChanged(TObject *Sender);
    void __fastcall OnMeshClick(TObject *Sender);
    void __fastcall OnSimSettingEdited(TObject *Sender, System::LongInt ACol,
                                       System::LongInt ARow,
                                       const System::UnicodeString Value);
	void __fastcall SaveAs2Click(TObject *Sender);
    void __fastcall OnFileOpen(TObject *Sender);
    void __fastcall OnFileSave(TObject *Sender);
    void __fastcall OnFileSaveAs(TObject *Sender);
    void __fastcall OnFileNew(TObject *Sender);

private:        // User declarations
    // 3D viewport: custom OpenGL control (not IDE-registered), created at
    // runtime inside the pnlView placeholder panel from the DFM
    TGLView       *glView         = nullptr;

    // scene
    struct ObjEntry
    {
        SceneObject obj;
        TriMesh     stlBase;     // unscaled mesh for imported STL
        bool        isStl = false;
    };
    std::vector<ObjEntry> objects;

    // simulation (time-domain solvers run through the common interface)
    std::unique_ptr<IFieldSolver> solver;
    std::unique_ptr<FemSolver> femSolver;
    std::unique_ptr<MomSolver> momSolver;
    std::unique_ptr<MomSurface> momSurf;
    bool                       usingFem      = false;
    bool                       usingMom      = false;
    bool                       usingMomSurf  = false;
    bool                       usingSweep    = false;
    std::thread                solverThread;
    bool                       threadJoined  = true;
    bool                       dftLoaded     = false;
    bool                       applyingProject = false;  // suppress combo handlers during load
    VoxelGridSpec              lastGrid;
    bool                       haveGrid      = false;

    // analysis / recording
    TChartForm   *chartForm   = nullptr;
    GifWriter     gif;
    bool          recordingGif = false;
    bool          gifStarted   = false;
    std::wstring  gifPath;
    int           lastGifStep  = -1;

    // playback
    bool           playbackMode   = false;
    bool           playing        = false;
    bool           updatingSlider = false;

    // loaded playback store: set when a saved result is displayed with no live
    // solver, so the frame scrubber replays recorded frames from a .emsim file
    bool                     usingLoaded = false;
    std::vector<VizFrame>    loadedFrames;
    std::vector<SurfaceFace> loadedFaces;
    float                    loadedDt    = 0.0f;

    // helpers
    void refreshObjectList(int select);
    void deleteSelectedObject();
    void refreshPropEditor();
    void updateSceneView();
    double simValue(const String &key, double def);
    void setSimValue(const String &key, const String &v);
    void addAntenna(AntennaKind kind);
    int  selectedIndex();
    void startSimulation();
    void stopSimulation(bool wait);
    void finishThread();
    void applyProperties();
    void updateVisualization();
    void invalidateResults();
    void showPlots();
    void stopGifRecording();
    void captureGifFrame();
    void showPlaybackFrame(int idx);
    void resetPlayback(bool enable);
    bool computeGridPreview(VoxelGridSpec &g);
    void updatePwMarker();
    void finishFemRun();
    void finishMomRun();
    void finishMomSurfRun();
    void finishSweep();
    void updateMeshView();
    void saveProjectTo(const String &path);
    bool loadProjectFrom(const String &path);
    String projectPath;
    // playback source abstraction (live solver or loaded snapshot)
    int   playbackFrameCount();
    bool  playbackGetFrame(int idx, VizFrame &out);
    const std::vector<SurfaceFace> &playbackFaces();
    float playbackDt();
    bool  playbackReady();
    void voxelizeScene(const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                       std::vector<MatProps> &table);
    bool meshPreviewShown = false;

public:         // User declarations
    __fastcall TMainForm(TComponent* Owner);
    __fastcall virtual ~TMainForm();
};
//---------------------------------------------------------------------------
extern PACKAGE TMainForm *MainForm;
//---------------------------------------------------------------------------
#endif
