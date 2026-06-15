object MainForm: TMainForm
  Left = 0
  Top = 0
  Caption = 'RF Simulator - 3D TLM Solver'
  ClientHeight = 860
  ClientWidth = 1380
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  Menu = MainMenu1
  Position = poScreenCenter
  WindowState = wsMaximized
  OnClose = OnFormClose
  TextHeight = 15
  object splitter: TSplitter
    Left = 300
    Top = 105
    Width = 5
    Height = 683
    ExplicitTop = 96
    ExplicitHeight = 692
  end
  object topPanel: TPanel
    Left = 0
    Top = 23
    Width = 1380
    Height = 82
    Align = alTop
    BevelOuter = bvNone
    ShowCaption = False
    TabOrder = 0
    object lblComponent: TLabel
      Left = 8
      Top = 6
      Width = 67
      Height = 15
      Caption = 'Component:'
    end
    object lblExcitation: TLabel
      Left = 180
      Top = 6
      Width = 54
      Height = 15
      Caption = 'Excitation:'
    end
    object lblSolver: TLabel
      Left = 426
      Top = 6
      Width = 35
      Height = 15
      Caption = 'Solver:'
    end
    object lblDisplay: TLabel
      Left = 609
      Top = 6
      Width = 41
      Height = 15
      Caption = 'Display:'
    end
    object cbAddKind: TComboBox
      Left = 8
      Top = 24
      Width = 160
      Height = 23
      Style = csDropDownList
      TabOrder = 0
    end
    object cbExcitation: TComboBox
      Left = 180
      Top = 24
      Width = 115
      Height = 23
      Style = csDropDownList
      ItemIndex = 0
      TabOrder = 1
      Text = 'Wire port(s)'
      OnChange = OnExcitationChanged
      Items.Strings = (
        'Wire port(s)'
        'Plane wave')
    end
    object cbWaveform: TComboBox
      Left = 299
      Top = 24
      Width = 115
      Height = 23
      Style = csDropDownList
      ItemIndex = 0
      TabOrder = 2
      Text = 'CW (ramped)'
      Items.Strings = (
        'CW (ramped)'
        'Gaussian sine'
        'Gaussian pulse')
    end
    object cbSolver: TComboBox
      Left = 426
      Top = 24
      Width = 175
      Height = 23
      Style = csDropDownList
      ItemIndex = 0
      TabOrder = 3
      Text = 'Solver: TLM (time domain)'
      Items.Strings = (
        'Solver: TLM (time domain)'
        'Solver: FDTD (time domain)'
        'Solver: FEM (frequency)'
        'Solver: MoM (wire)'
        'Solver: MoM (surface)')
    end
    object cbCurrents: TComboBox
      Left = 609
      Top = 24
      Width = 135
      Height = 23
      Style = csDropDownList
      ItemIndex = 1
      TabOrder = 4
      Text = 'Currents: live'
      OnChange = OnViewOptionChanged
      Items.Strings = (
        'Currents: off'
        'Currents: live'
        'Currents: Js @ f0')
    end
    object cbDbRange: TComboBox
      Left = 748
      Top = 24
      Width = 120
      Height = 23
      Style = csDropDownList
      ItemIndex = 2
      TabOrder = 5
      Text = 'Range: 40 dB'
      OnChange = OnViewOptionChanged
      Items.Strings = (
        'Range: 20 dB'
        'Range: 30 dB'
        'Range: 40 dB'
        'Range: 50 dB'
        'Range: 60 dB')
    end
    object chkModel: TCheckBox
      Left = 884
      Top = 6
      Width = 70
      Height = 17
      Caption = 'Model'
      Checked = True
      State = cbChecked
      TabOrder = 6
      OnClick = OnViewOptionChanged
    end
    object chkDb: TCheckBox
      Left = 884
      Top = 28
      Width = 80
      Height = 17
      Caption = 'dB scale'
      Checked = True
      State = cbChecked
      TabOrder = 7
      OnClick = OnViewOptionChanged
    end
    object chkPlane: TCheckBox
      Left = 964
      Top = 6
      Width = 110
      Height = 17
      Caption = '|E| cut plane'
      TabOrder = 8
      OnClick = OnViewOptionChanged
    end
    object chkMesh: TCheckBox
      Left = 964
      Top = 28
      Width = 100
      Height = 17
      Caption = 'FEM mesh'
      TabOrder = 9
      OnClick = OnViewOptionChanged
    end
    object chkGpu: TCheckBox
      Left = 8
      Top = 53
      Width = 150
      Height = 17
      Caption = 'Use GPU (compute)'
      TabOrder = 10
    end
  end
  object leftPanel: TPanel
    Left = 0
    Top = 105
    Width = 300
    Height = 683
    Align = alLeft
    BevelOuter = bvNone
    ShowCaption = False
    TabOrder = 1
    ExplicitTop = 74
    ExplicitHeight = 697
    object lblComponents: TLabel
      Left = 0
      Top = 0
      Width = 300
      Height = 15
      Align = alTop
      Caption = ' Components'
      ExplicitWidth = 72
    end
    object lblProps: TLabel
      Left = 0
      Top = 145
      Width = 300
      Height = 15
      Align = alTop
      Caption = ' Properties (selected)'
      ExplicitWidth = 110
    end
    object lblSim: TLabel
      Left = 0
      Top = 418
      Width = 300
      Height = 15
      Align = alTop
      Caption = ' Simulation settings'
      ExplicitWidth = 104
    end
    object lbObjects: TListBox
      Left = 0
      Top = 15
      Width = 300
      Height = 130
      Align = alTop
      ItemHeight = 15
      PopupMenu = pmObjects
      TabOrder = 0
      OnClick = OnObjectSelect
      OnKeyDown = OnObjectsKeyDown
    end
    object vleProps: TValueListEditor
      Left = 0
      Top = 160
      Width = 300
      Height = 230
      Align = alTop
      TabOrder = 1
      TitleCaptions.Strings = (
        'Property'
        'Value')
      ColWidths = (
        150
        144)
    end
    object btnApply: TButton
      Left = 0
      Top = 390
      Width = 300
      Height = 28
      Align = alTop
      Caption = 'Apply properties'
      TabOrder = 2
      OnClick = OnApplyClick
    end
    object vleSim: TValueListEditor
      Left = 0
      Top = 433
      Width = 300
      Height = 250
      Align = alClient
      Strings.Strings = (
        'Frequency (MHz)=1000'
        'Cells per lambda=20'
        'Padding (cells)=12'
        'Timesteps (0:auto)=0'
        'Settle (0:auto)=0'
        'Boundary rho=0'
        'PW prop axis (0..2)=0'
        'PW pol axis (0..2)=2'
        'Cut plane axis=1'
        'RCS sweep f1 (MHz)=10'
        'RCS sweep f2 (MHz)=40'
        'RCS sweep points=16')
      TabOrder = 3
      TitleCaptions.Strings = (
        'Setting'
        'Value')
      OnSetEditText = OnSimSettingEdited
      ExplicitHeight = 264
      ColWidths = (
        150
        144)
    end
  end
  object pnlView: TPanel
    Left = 305
    Top = 105
    Width = 1075
    Height = 683
    Align = alClient
    BevelOuter = bvNone
    ShowCaption = False
    TabOrder = 2
    ExplicitTop = 160
    ExplicitHeight = 628
  end
  object playPanel: TPanel
    Left = 0
    Top = 788
    Width = 1380
    Height = 36
    Align = alBottom
    BevelOuter = bvNone
    ShowCaption = False
    TabOrder = 3
    ExplicitTop = 771
    ExplicitWidth = 1374
    object lblPlayback: TLabel
      Left = 0
      Top = 0
      Width = 59
      Height = 36
      Align = alLeft
      Caption = '  Playback: '
      Layout = tlCenter
      ExplicitHeight = 15
    end
    object lblFrame: TLabel
      Left = 1120
      Top = 0
      Width = 260
      Height = 36
      Align = alRight
      AutoSize = False
      Caption = 'run a simulation to record frames  '
      Layout = tlCenter
    end
    object btnPlay: TButton
      Left = 59
      Top = 0
      Width = 70
      Height = 36
      Align = alLeft
      Caption = #9654' Play'
      Enabled = False
      TabOrder = 0
      OnClick = OnPlayClick
    end
    object tbPlayback: TTrackBar
      Left = 129
      Top = 0
      Width = 991
      Height = 36
      Align = alClient
      Enabled = False
      Max = 0
      TabOrder = 1
      TickStyle = tsNone
      OnChange = OnPlaybackChange
      ExplicitWidth = 985
    end
  end
  object statusBar: TStatusBar
    Left = 0
    Top = 824
    Width = 1380
    Height = 36
    Panels = <
      item
        Width = 500
      end
      item
        Width = 180
      end
      item
        Width = 180
      end
      item
        Width = 180
      end>
    ExplicitTop = 807
    ExplicitWidth = 1374
  end
  object ToolBar1: TToolBar
    Left = 0
    Top = 0
    Width = 1380
    Height = 23
    AutoSize = True
    ButtonHeight = 23
    ButtonWidth = 124
    Caption = 'ToolBar1'
    ShowCaptions = True
    TabOrder = 5
    ExplicitWidth = 1374
    object btnAdd: TToolButton
      Left = 0
      Top = 0
      AutoSize = True
      Caption = 'Add'
      OnClick = OnAddClick
    end
    object btnImport: TToolButton
      Left = 33
      Top = 0
      AutoSize = True
      Caption = 'Import STL...'
      OnClick = OnImportClick
    end
    object btnDelete: TToolButton
      Left = 111
      Top = 0
      AutoSize = True
      Caption = 'Delete'
      OnClick = OnDeleteClick
    end
    object ToolSep1: TToolButton
      Left = 155
      Top = 0
      Width = 8
      Style = tbsSeparator
    end
    object btnMesh: TToolButton
      Left = 163
      Top = 0
      AutoSize = True
      Caption = 'Generate mesh'
      OnClick = OnMeshClick
    end
    object ToolSep2: TToolButton
      Left = 253
      Top = 0
      Width = 8
      Style = tbsSeparator
    end
    object btnRun: TToolButton
      Left = 261
      Top = 0
      AutoSize = True
      Caption = #9654' Run'
      OnClick = OnRunClick
    end
    object btnStop: TToolButton
      Left = 306
      Top = 0
      AutoSize = True
      Caption = #9632' Stop'
      Enabled = False
      OnClick = OnStopClick
    end
    object ToolSep3: TToolButton
      Left = 354
      Top = 0
      Width = 8
      Style = tbsSeparator
    end
    object btnPlots: TToolButton
      Left = 362
      Top = 0
      AutoSize = True
      Caption = 'Plots (S11/Z/energy)...'
      OnClick = OnPlotsClick
    end
    object btnFarField: TToolButton
      Left = 490
      Top = 0
      AutoSize = True
      Caption = 'Far-field pattern'
      OnClick = OnFarFieldClick
    end
    object btnRecGif: TToolButton
      Left = 586
      Top = 0
      AutoSize = True
      Caption = #9679' Record GIF...'
      OnClick = OnRecGifClick
    end
    object ToolSep4: TToolButton
      Left = 673
      Top = 0
      Width = 8
      Style = tbsSeparator
    end
    object btnZoom: TToolButton
      Left = 681
      Top = 0
      AutoSize = True
      Caption = 'Zoom extents'
      OnClick = OnZoomClick
    end
  end
  object uiTimer: TTimer
    Interval = 100
    OnTimer = OnTimerTick
    Left = 1304
    Top = 120
  end
  object MainMenu1: TMainMenu
    Left = 328
    Top = 112
    object File1: TMenuItem
      Caption = 'File'
      object Close1: TMenuItem
        Caption = 'New'
        OnClick = OnFileNew
      end
      object File2: TMenuItem
        Caption = 'Open...'
        OnClick = OnFileOpen
      end
      object N2: TMenuItem
        Caption = '-'
      end
      object Close2: TMenuItem
        Caption = 'Save'
        OnClick = OnFileSave
      end
      object SaveAs1: TMenuItem
        Caption = 'Save As...'
        OnClick = OnFileSaveAs
      end
      object N1: TMenuItem
        Caption = '-'
      end
      object SaveAs2: TMenuItem
        Caption = 'Exit'
        OnClick = SaveAs2Click
      end
    end
  end
  object pmObjects: TPopupMenu
    Left = 120
    Top = 112
    object miDelete: TMenuItem
      Caption = 'Delete'
      OnClick = OnObjectDelete
    end
  end
end
