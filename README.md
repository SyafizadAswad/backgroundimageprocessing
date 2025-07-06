# 🇯🇵 GPU最適化画像ブランディングツール

Streamlitベースの日本語画像ブランディングアプリ（GPU対応）です。商品画像の背景除去、ロゴ・テキスト追加、1000x1000pxへのリサイズなどが簡単に行えます。

---

## 🚀 セットアップガイド

本アプリは、以下のプログラムが必須となりますので、
事前にインストールしてください。
・Python 3.7+
・pip

### 1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. 仮想環境の作成（推奨）
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 必要なパッケージのインストール
```bash
pip install -r requirements_gpu.txt
```
> **GPUサポート**  
> requirements_gpu.txt には PyTorch（CUDA対応）などGPU用パッケージが含まれています。  
> NVIDIA GPUと対応ドライバが必要です。

### 4. フォントファイルの確認
`NotoSansJP-Regular.ttf` がプロジェクトディレクトリにあることを確認してください。  
このフォントは日本語テキストの描画に使用されます。

### 5. U^2-Netモデルのダウンロード（初回のみ）
背景除去には `backgroundremover` パッケージがU^2-Netモデルを使用します。  
初回実行時に自動ダウンロードされますが、失敗する場合は手動で  
https://github.com/xuebinqin/U-2-Net/releases  
から `u2net.pth` をダウンロードし、ホームディレクトリの `.u2net` フォルダに配置してください。

### 6. アプリの起動
```bash
streamlit run imageprocessing_JP.py
```

### 7. ブラウザでアクセス
起動後、Streamlitが表示するURL（例: http://localhost:8501）をブラウザで開いてください。

---

## 📝 必要ファイル一覧
- `imageprocessing_JP.py`（アプリ本体）
- `requirements_gpu.txt`（依存パッケージリスト）
- `NotoSansJP-Regular.ttf`（日本語フォント）

---

## 🛠️ トラブルシューティング
- 日本語が□で表示される場合は、`NotoSansJP-Regular.ttf` が存在するか確認してください。
- GPUが認識されない場合は、NVIDIAドライバやCUDA、PyTorchのバージョンを確認してください。
- 背景除去が動作しない場合は、U^2-Netモデルファイルの配置を確認してください。

---

## 🔗 参考
- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [Noto Sans JPフォント](https://fonts.google.com/noto/specimen/Noto+Sans+JP)
- [U^2-Netモデル](https://github.com/xuebinqin/U-2-Net/releases)

---

## 📋 実行スクリプト

### 1. 仮想環境のアクティベートとアプリの起動

`run_imageprocessing_JP.bat` を使用して、仮想環境をアクティベートし、アプリを起動します。

**使い方:**
1. このスクリプトを `run_imageprocessing_JP.bat` としてプロジェクトフォルダに保存します。
2. ダブルクリックで実行します。
   (仮想環境がアクティベートされ、アプリが起動します。)

---

ご利用ありがとうございます！不明点や要望があればIssueやPRでご連絡ください。 
