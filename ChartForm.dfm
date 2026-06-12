object ChartForm: TChartForm
  Left = 0
  Top = 0
  Caption = 'Plots'
  ClientHeight = 560
  ClientWidth = 900
  Color = clWhite
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -12
  Font.Name = 'Segoe UI'
  Font.Style = []
  Position = poScreenCenter
  TextHeight = 15
  object paintBox: TPaintBox
    Left = 0
    Top = 36
    Width = 900
    Height = 524
    Align = alClient
    OnPaint = onPaint
  end
  object topBar: TPanel
    Left = 0
    Top = 0
    Width = 900
    Height = 36
    Align = alTop
    BevelOuter = bvNone
    Color = clWhite
    ParentBackground = False
    ShowCaption = False
    TabOrder = 0
    object cbPage: TComboBox
      Left = 8
      Top = 6
      Width = 260
      Height = 23
      Style = csDropDownList
      TabOrder = 0
      OnChange = onPageChange
    end
    object btnCsv: TButton
      Left = 276
      Top = 5
      Width = 100
      Height = 25
      Caption = 'Export CSV...'
      TabOrder = 1
      OnClick = onCsvClick
    end
  end
end
