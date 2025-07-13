# -*- coding: utf-8 -*-
# GPU最適化画像処理アプリ

# imageprocessing_JP.py
#
# このスクリプトは、Streamlitを使用して画像を処理するローカルWebアプリケーションを作成します。
# ユーザーは以下ができます：
#  1. メイン画像とオプションのロゴをアップロード。
#  2. 利用可能な場合はGPUアクセラレーションを使用してメイン画像から背景を削除。
#  3. ロゴとカスタムテキストを4つの角のいずれかに追加。
#  4. 最終画像を1000x1000ピクセルにリサイズし、アスペクト比を保持して
#     白い背景を追加。
#
# GPU最適化機能：
# - 自動GPU検出と利用
# - GPUが利用できない場合はCPUにフォールバック
# - 高速化のための処理前画像リサイズ
# - 進捗インジケーターとパフォーマンス指標

# 要件: streamlit, pillow, backgroundremover, torch
# インストール: pip install streamlit pillow backgroundremover torch torchvision
# GPUサポート: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from backgroundremover.bg import remove
import io
import time
import torch
import os
from streamlit_paste_button import paste_image_button

# --- 設定と定数 ---
OUTPUT_SIZE = 1000
LOGO_MAX_SIZE = 180
PADDING = 40
MAX_INPUT_SIZE = 800  # 高速化のための処理前画像リサイズ

# --- GPU/CPU検出とセットアップ ---

def check_gpu_availability():
    """GPUが利用可能かどうかをチェックし、デバイス情報を返す。"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, f"GPU: {device_name} ({memory_gb:.1f}GB)"
    else:
        return False, "CPUのみ"

def resize_image_for_processing(image_bytes, max_size=MAX_INPUT_SIZE):
    """高速処理のため、アスペクト比を維持しながら画像をmax_sizeにリサイズ。"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # リサイズが必要かチェック
        if max(image.size) <= max_size:
            return image_bytes
        
        # アスペクト比を維持しながらリサイズ
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # バイトに変換して戻す
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        st.warning(f"画像のリサイズができませんでした: {e}")
        return image_bytes

# --- コア画像処理関数 ---

def crop_and_center_object(image, canvas_size=1000, padding=10, alpha_threshold=200):
    import numpy as np
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        non_transparent = np.where(alpha > alpha_threshold)
    else:
        alpha = np.ones((img_array.shape[0], img_array.shape[1])) * 255
        non_transparent = np.where(alpha > 0)
    if len(non_transparent[0]) == 0:
        # fallback: center the whole image
        final_canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
        image.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)
        paste_x = (canvas_size - image.width) // 2
        paste_y = (canvas_size - image.height) // 2
        final_canvas.paste(image, (paste_x, paste_y), image)
        return final_canvas
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
    # Expand the box by padding, but keep within image bounds
    min_y = max(0, min_y - padding)
    max_y = min(image.height - 1, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(image.width - 1, max_x + padding)
    # Crop the object
    cropped = image.crop((min_x, min_y, max_x + 1, max_y + 1))
    # Create a new white canvas
    final_canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    # Resize cropped object if it's too big
    cropped.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    # Center the cropped object
    paste_x = (canvas_size - cropped.width) // 2
    paste_y = (canvas_size - cropped.height) // 2
    final_canvas.paste(cropped, (paste_x, paste_y), cropped)
    return final_canvas

def process_image(main_image_bytes, logo_image_bytes, text_inputs, logo_position, text_positions, use_gpu=True):
    """
    画像を処理するメイン関数：背景削除、リサイズ、ブランディング追加。
    
    Args:
        main_image_bytes (bytes): 処理するメイン画像のバイトデータ。
        logo_image_bytes (bytes): ロゴ画像のバイトデータ。Noneの場合もある。
        text_inputs (list): 'text'と'position'キーを持つ辞書のリスト。
        logo_position (str): ロゴを配置する角。
        text_positions (list): 各テキスト要素の位置のリスト。
        use_gpu (bool): 利用可能な場合はGPUアクセラレーションを使用するかどうか。

    Returns:
        PIL.Image.Image: 最終処理済み画像オブジェクト。
    """
    try:
        # 処理情報を表示
        gpu_available, device_info = check_gpu_availability()
        if use_gpu and gpu_available:
            st.info(f"{device_info}で処理中")
        else:
            st.info(f"CPUで処理中")
        
        # ステップ1: 高速処理のため画像をリサイズ
        with st.spinner('最適な処理のため画像をリサイズ中...'):
            resized_image_bytes = resize_image_for_processing(main_image_bytes)
        
        # ステップ2: メイン画像から背景を削除。
        with st.spinner('背景を削除中（10-30秒かかる場合があります）...'):
            start_time = time.time()
            
            # `remove`関数はバイトを受け取り、アルファチャンネル付きPNGのバイトを返す。
            foreground_bytes = remove(resized_image_bytes)
            
            processing_time = time.time() - start_time
            st.success(f"背景削除完了（{processing_time:.1f}秒）")
        
        foreground_image = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")

        # ステップ3: 白い背景の新しい空白キャンバスを作成。
        # 透明な貼り付けを可能にするためRGBAを使用。
        # final_canvas = Image.new("RGBA", (OUTPUT_SIZE, OUTPUT_SIZE), (255, 255, 255, 255))

        # ステップ4: アスペクト比を維持しながら、キャンバス内に収まるように前景画像をリサイズ。
        # `thumbnail`メソッドは画像をその場でリサイズする。
        foreground_copy = foreground_image.copy()
        foreground_copy.thumbnail((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)

        # --- Improved centering: crop to object and center on canvas ---
        final_canvas = crop_and_center_object(foreground_copy, OUTPUT_SIZE, padding=10, alpha_threshold=200)

        # --- Continue with logo and text as before ---
        # (Removed old centering and pasting logic)

        # ステップ7: ロゴが提供されている場合は追加。
        if logo_image_bytes:
            with st.spinner('ロゴを追加中...'):
                logo_image = Image.open(io.BytesIO(logo_image_bytes)).convert("RGBA")
                logo_image.thumbnail((LOGO_MAX_SIZE, LOGO_MAX_SIZE), Image.Resampling.LANCZOS)
                
                logo_x, logo_y = calculate_position(
                    logo_position,
                    logo_image.width,
                    logo_image.height,
                    is_logo=True
                )
                final_canvas.paste(logo_image, (logo_x, logo_y), logo_image)

        # ステップ8: テキストが提供されている場合は追加。
        if text_inputs:
            with st.spinner('テキストを追加中...'):
                draw = ImageDraw.Draw(final_canvas)
                # 常にプロジェクトディレクトリのNotoSansJP-Regular.ttfフォントを使用
                font_path = os.path.join(os.path.dirname(__file__), "NotoSansJP-Regular.ttf")
                try:
                    font = ImageFont.truetype(font_path, size=32)
                except Exception as e:
                    font = ImageFont.load_default()
                
                # 各テキスト要素を追加
                for i, text_data in enumerate(text_inputs):
                    if text_data['text'].strip():  # 空でないテキストのみ追加
                        text_to_add = text_data['text']
                        position = text_data['position']
                        
                        # 改行でテキストを分割し、ワードラップを処理
                        lines = text_to_add.split('\n')
                        wrapped_lines = []
                        max_width = OUTPUT_SIZE - 2 * PADDING  # テキストの最大幅
                        
                        for line in lines:
                            if line.strip():
                                # 行が長すぎる場合はワードラップ
                                words = line.split()
                                current_line = ""
                                for word in words:
                                    test_line = current_line + " " + word if current_line else word
                                    bbox = draw.textbbox((0, 0), test_line, font=font)
                                    if bbox[2] - bbox[0] <= max_width:
                                        current_line = test_line
                                    else:
                                        if current_line:
                                            wrapped_lines.append(current_line)
                                        current_line = word
                                if current_line:
                                    wrapped_lines.append(current_line)
                            else:
                                wrapped_lines.append("")  # 間隔のための空行
                        
                        # すべてのラップされた行の合計高さを計算
                        line_height = font.getbbox("Ay")[3]  # 概算の行の高さ
                        total_height = len(wrapped_lines) * line_height
                        
                        # 最初の行の位置を計算
                        text_x, text_y = calculate_position(
                            position,
                            max_width,  # 位置決めに最大幅を使用
                            total_height,
                            is_logo=False,
                            logo_height=logo_image.height if logo_image_bytes else 0
                        )
                        
                        # 各行を描画
                        for line in wrapped_lines:
                            if line.strip():  # 空でない行のみ描画
                                if position == 'Bottom Center':
                                    bbox = draw.textbbox((0, 0), line, font=font)
                                    text_width = bbox[2] - bbox[0]
                                    adjusted_x = (OUTPUT_SIZE - text_width) // 2
                                elif 'Right' in position:
                                    bbox = draw.textbbox((0, 0), line, font=font)
                                    text_width = bbox[2] - bbox[0]
                                    adjusted_x = text_x + (max_width - text_width)
                                else:
                                    adjusted_x = text_x
                                draw.text((adjusted_x, text_y), line, font=font, fill=(0, 0, 0, 255))
                            text_y += line_height

        # より広い互換性のためRGBに変換（例：JPEGとして保存）。
        return final_canvas.convert("RGB")

    except Exception as e:
        st.error(f"画像処理中にエラーが発生しました: {e}")
        print(f"画像処理エラー: {e}")
        return None

def center_product_content(image, canvas_size):
    """
    画像内の実際の商品内容（非透明部分）を検出し、キャンバス内で適切に中央配置する座標を計算する。
    
    Args:
        image (PIL.Image): RGBA画像（背景除去済み）
        canvas_size (int): キャンバスのサイズ（正方形）
    
    Returns:
        tuple: (x, y) 座標
    """
    # 画像をnumpy配列に変換してアルファチャンネルを取得
    import numpy as np
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # RGBA画像の場合、アルファチャンネルを使用
        alpha = img_array[:, :, 3]
        threshold = 250 # より厳しめの閾値で半透明部分を無視
        non_transparent = np.where(alpha > threshold)
    else:
        # RGB画像の場合、すべてのピクセルを不透明として扱う
        alpha = np.ones((img_array.shape[0], img_array.shape[1])) * 255
        non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:
        # 透明ピクセルしかない場合は、画像全体を中央に配置
        return (canvas_size - image.width) // 2, (canvas_size - image.height) // 2
    
    # 非透明部分の境界を計算
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()
    
    # 実際の商品内容の中心を計算
    content_center_x = (min_x + max_x) // 2
    content_center_y = (min_y + max_y) // 2
    
    # キャンバスの中心
    canvas_center = canvas_size // 2
    
    # 商品内容の中心がキャンバスの中心に来るように画像全体を移動
    offset_x = canvas_center - content_center_x
    offset_y = canvas_center - content_center_y
    
    # 画像がキャンバスからはみ出さないように調整
    final_x = max(0, min(offset_x, canvas_size - image.width))
    final_y = max(0, min(offset_y, canvas_size - image.height))
    
    return final_x, final_y

def calculate_position(position, item_width, item_height, is_logo, logo_height=0):
    """
    選択された角に基づいてアイテムの(x, y)座標を計算する。
    下部の場合は常にキャンバスの端に固定されるように修正。
    """
    if position == 'Bottom Center':
        x = (OUTPUT_SIZE - item_width) // 2
        y = OUTPUT_SIZE - item_height - PADDING
    elif 'Right' in position:
        x = OUTPUT_SIZE - item_width - PADDING
        y = OUTPUT_SIZE - item_height - PADDING if 'Bottom' in position else PADDING
    elif 'Left' in position:
        x = PADDING
        y = OUTPUT_SIZE - item_height - PADDING if 'Bottom' in position else PADDING
    else:  # Default to Top
        x = (OUTPUT_SIZE - item_width) // 2
        y = PADDING
    return x, y

# --- Streamlitユーザーインターフェース ---

st.set_page_config(layout="wide", page_title="GPU最適化画像ブランディングツール")

st.title("GPU最適化画像ブランディングツール")
st.markdown("""
このツールは、GPUアクセラレーションを使用してオンラインストアやソーシャルメディア用の画像を準備するのに役立ちます。
1.  **商品画像をアップロード**して背景を自動削除。
2.  オプションで、**ロゴをアップロード**し、**テキストを追加**してブランディング。
3.  最終画像は、白い背景のきれいな**1000x1000px**の正方形になります。
""")

# GPU/CPUステータスをチェックして表示
gpu_available, device_info = check_gpu_availability()
if gpu_available:
    st.success(f"🎉 {device_info}が検出され、アクセラレーションの準備ができました！")
else:
    st.info(f"💻 {device_info} - 処理はCPUを使用します")

# 結果を保持するセッション状態を初期化
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

col1, col2 = st.columns(2)

with col1:
    st.header("⚙️ 入力")
    st.subheader("1. メイン画像をアップロード・貼り付け")
    # Paste from clipboard button
    pasted_image = paste_image_button("クリップボードから画像を貼り付け")
    
    # Show a mini preview if an image was pasted
    if pasted_image is not None:
        img = getattr(pasted_image, 'data', None) or getattr(pasted_image, 'image_data', None)
        if img is not None:
            st.image(img, caption="貼り付けた画像プレビュー", width=200)
    
    # File uploader fallback
    main_image_file = st.file_uploader(
        "画像をアップロード（ドラッグ＆ドロップまたはファイル選択）",
        type=['png', 'jpg', 'jpeg', 'webp']
    )

    # Use whichever is provided (priority: pasted > file)
    main_image_bytes = None
    if pasted_image is not None:
        import io
        buf = io.BytesIO()
        img = getattr(pasted_image, 'data', None) or getattr(pasted_image, 'image_data', None)
        if img is not None:
            img.save(buf, format="PNG")
            main_image_bytes = buf.getvalue()
    elif main_image_file is not None:
        main_image_bytes = main_image_file.getvalue()

    logo_image_file = st.file_uploader(
        "2. ロゴをアップロード（オプション）",
        type=['png', 'jpg', 'jpeg', 'webp']
    )
    
    # ロゴの位置決め
    if logo_image_file:
        logo_position = st.selectbox(
            "3. ロゴの位置を選択",
            ('Top Right', 'Top Left', 'Bottom Right', 'Bottom Left'),
            key="logo_position"
        )
    
    # テキスト入力セクション
    st.subheader("4. テキストを追加（オプション）")
    
    # セッション状態でテキスト入力を初期化
    if 'text_inputs' not in st.session_state:
        st.session_state.text_inputs = [{'text': '', 'position': 'Top Right'}]
    
    # テキスト入力フィールドを追加
    for i, text_data in enumerate(st.session_state.text_inputs):
        col_text, col_pos = st.columns([3, 1])
        with col_text:
            st.session_state.text_inputs[i]['text'] = st.text_area(
                f"テキスト {i+1}",
                value=text_data['text'],
                placeholder="ここにテキストを入力...\n改行するにはEnterキーを使用",
                height=100,
                key=f"text_input_{i}"
            )
        with col_pos:
            st.session_state.text_inputs[i]['position'] = st.selectbox(
                "位置",
                ('Top Right', 'Top Left', 'Bottom Right', 'Bottom Left', 'Bottom Center'),
                index=('Top Right', 'Top Left', 'Bottom Right', 'Bottom Left', 'Bottom Center').index(text_data['position']),
                key=f"text_position_{i}"
            )
    
    # テキスト追加/削除ボタン
    col_add, col_remove = st.columns(2)
    with col_add:
        if st.button("テキストを追加"):
            st.session_state.text_inputs.append({'text': '', 'position': 'Top Right'})
            st.rerun()
    
    with col_remove:
        if len(st.session_state.text_inputs) > 1 and st.button("最後のテキストを削除"):
            st.session_state.text_inputs.pop()
            st.rerun()
    
    # ダウンロード用のカスタムファイル名入力（処理前）
    st.subheader("5. ダウンロード用ファイル名")
    default_filename = "processed_image"
    custom_filename = st.text_input(
        "ダウンロード用ファイル名を入力（拡張子なし）:",
        value=default_filename,
        help="ファイルはPNG画像として保存されます"
    )
    
    # ファイル名をクリーンアップ（無効な文字を削除）
    import re
    clean_filename = re.sub(r'[<>:"/\\|?*]', '_', custom_filename.strip())
    if not clean_filename:
        clean_filename = default_filename
    
    # GPU/CPU選択
    if gpu_available:
        use_gpu = st.checkbox("GPUアクセラレーションを使用（高速）", value=True)
    else:
        use_gpu = False
        st.info("GPUが利用できません - CPUを使用します")

    process_button = st.button("✨ 画像を処理", type="primary")

with col2:
    st.header("✅ 結果")
    if process_button and main_image_bytes:
        with st.spinner('処理中... しばらくお待ちください。'):
            logo_image_bytes = logo_image_file.getvalue() if logo_image_file else None
            
            # ロゴ位置を取得（ロゴがない場合はNone）
            logo_pos = logo_position if logo_image_file else None
            
            # 空のテキスト入力をフィルタリング
            valid_text_inputs = [text_data for text_data in st.session_state.text_inputs if text_data['text'].strip()]
            
            # 結果をセッション状態に保存
            st.session_state.processed_image = process_image(
                main_image_bytes, logo_image_bytes, valid_text_inputs, logo_pos, [], use_gpu
            )
            # ダウンロード用ファイル名を保存
            st.session_state.download_filename = clean_filename

    # セッション状態から画像を表示
    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="処理済み画像", use_column_width=True)
        
        # ダウンロードボタン用にPIL画像をバイトに変換
        buf = io.BytesIO()
        st.session_state.processed_image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # 保存されたファイル名またはデフォルトを使用
        download_filename = getattr(st.session_state, 'download_filename', 'processed_image')
        
        st.download_button(
            label="📥 画像をダウンロード",
            data=img_bytes,
            file_name=f"{download_filename}.png",
            mime="image/png"
        )
    else:
        st.info("画像をアップロードして「画像を処理」をクリックすると、ここに結果が表示されます。")

