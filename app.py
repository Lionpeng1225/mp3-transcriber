import streamlit as st
from pathlib import Path
import os
import importlib.util
from speaker_token_store import load_token, save_token, delete_token
from transcription_job import (TERMINAL, TranscriptionJob, export_rows, timestamp,
                               register_job, get_job, saved_jobs, delete_job)


def job_snapshot(job):
    if hasattr(job, "content_revision"):
        return job.snapshot(include_rows=False)
    # A hot reload can retain a job created by the previous module version.
    state = job.snapshot()
    state["row_count"] = len(state["rows"])
    return state

st.set_page_config(page_title="MP3／MP4 轉逐字稿", page_icon="🎙️")
st.title("MP3／MP4 轉逐字稿")
st.write("選擇錄音或影片，在本機辨識並下載繁體中文逐字稿。")
st.caption("首次使用各模型需連網下載。錄音每兩分鐘分段處理，時間標記依原始音檔累計。")
st.caption("可選擇切分兩段並行轉錄：尋找至少 0.5 秒、音量不高於 −35 dBFS 且最靠近中點的靜音，在附近切分並保留聲音與短重疊；找不到則整段辨識。")
if "job" not in st.session_state and st.query_params.get("job"):
    try:
        st.session_state.job = get_job(st.query_params["job"])
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"無法找回指定工作：{error}")
        del st.query_params["job"]
with st.expander("找回已保存的工作"):
    history = saved_jobs()
    if history:
        labels = {item[0]: f"{item[1]}（{item[0][:8]}）" for item in history}
        selected = st.selectbox("保存的工作", list(labels), format_func=labels.get)
        if st.button("開啟此工作"):
            try:
                st.session_state.job = get_job(selected)
                st.query_params["job"] = selected
                st.rerun()
            except (OSError, ValueError, KeyError) as error:
                st.error(f"讀取失敗：{error}")
        confirmed = st.checkbox(
            "確認永久刪除此工作的逐字稿、轉錄進度與保存音檔（不會刪除原始上傳來源或已下載的檔案）",
            key=f"delete_confirm_{selected}",
        )
        if st.button("刪除此保存工作", disabled=not confirmed):
            try:
                delete_job(selected)
                current = st.session_state.get("job")
                if current is not None and current.id == selected:
                    del st.session_state["job"]
                if st.query_params.get("job") == selected:
                    del st.query_params["job"]
                st.session_state["deleted_job_notice"] = "已刪除保存工作及其保存音檔。"
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"刪除失敗：{error}")
    else:
        st.caption("尚無保存的工作。")
token_path = Path(__file__).resolve().parent / ".speaker-token.dpapi"
if "speaker_token_global" not in st.session_state:
    try:
        st.session_state["speaker_token_global"] = load_token(token_path)
    except Exception:
        st.session_state["speaker_token_global"] = ""
with st.expander("說話者辨識權杖設定（所有工作共用）"):
    st.caption("記住後，新上傳及已保存的工作都會自動使用這個權杖。權杖以 Windows 使用者加密方式儲存在本機。")
    st.markdown("[前往 Hugging Face 同意模型條款並建立 Read 權杖](https://huggingface.co/pyannote/speaker-diarization-community-1)")
    st.text_input("Hugging Face Read 權杖", type="password", key="speaker_token_global")
    remember, forget = st.columns(2)
    if remember.button("記住權杖", disabled=not st.session_state["speaker_token_global"].strip()):
        try:
            save_token(token_path, st.session_state["speaker_token_global"].strip())
            st.success("已加密儲存在這台電腦；所有新工作會自動套用。")
        except Exception as error:
            st.error(f"無法儲存權杖：{error}")
    if forget.button("刪除已記住的權杖"):
        try:
            delete_token(token_path)
            st.session_state["speaker_token_global"] = ""
            st.success("已刪除儲存的權杖。")
            st.rerun()
        except Exception as error:
            st.error(f"無法刪除權杖：{error}")
st.caption("每次辨識回傳後會自動保存。原始錄音與進度保存在程式的 saved_jobs 資料夾。")
if "deleted_job_notice" in st.session_state:
    st.success(st.session_state.pop("deleted_job_notice"))
if getattr(st.session_state.get("job"), "deleted", False):
    del st.session_state["job"]
    st.query_params.pop("job", None)
job = st.session_state.get("job")
active = job is not None and job_snapshot(job)["status"] not in TERMINAL
recording = st.file_uploader("選擇 MP3 錄音或 MP4 影片", type=["mp3", "mp4"], disabled=active)
st.caption("MP4 會讀取第一條音軌，不辨識影片畫面或畫面中的字幕；不需先轉成 MP3。")
engine_name = st.selectbox("轉錄引擎", ["faster-whisper", "whisper.cpp"], disabled=active)
backend = {"engine": engine_name}
backend_ready = True
if engine_name == "whisper.cpp":
    candidates = [Path(os.environ.get(folder, "")) / "Buzz" / "_internal" / "buzz" / "whisper_cpp" / "whisper-cli.exe"
                  for folder in ("ProgramFiles(x86)", "ProgramFiles")]
    installed_cli = next((str(path) for path in candidates if path.is_file()), "")
    backend["cli"] = st.text_input("whisper-cli.exe 完整路徑", value=installed_cli, disabled=active)
    local_ggml = Path(__file__).resolve().parent / "models" / "whisper-cpp" / "ggml-small.bin"
    backend["model"] = st.text_input("Whisper.cpp ggml 模型完整路徑",
                                     value=str(local_ggml) if local_ggml.is_file() else "", disabled=active)
    backend["gpu"] = st.checkbox("使用 GPU（需相容的 Vulkan 引擎與驅動）", disabled=active)
    backend_ready = Path(backend["cli"]).is_file() and Path(backend["model"]).is_file()
    st.caption("請使用 Buzz 安裝目錄內的 whisper-cli.exe 與已下載的 ggml 模型；安裝包的 .bin 是分割資料，不是辨識模型。")
size = st.selectbox("辨識模型（Faster Whisper）", ["small", "medium", "large-v3", "turbo"], disabled=active or engine_name == "whisper.cpp")
st.caption("Turbo 採用精簡解碼器，適合測試速度與品質的平衡；正確率仍需對照錄音。新模型下載至程式的 models 資料夾。")
language = st.selectbox("錄音語言", ["中文", "自動偵測", "英文"], disabled=active)
mode = st.selectbox("轉錄模式（Faster Whisper）", ["快速", "自適應快速", "極速", "原始精細"], disabled=active or engine_name == "whisper.cpp")
if mode == "自適應快速" and engine_name == "faster-whisper":
    st.caption("先用單一路徑辨識。可能有人聲但結果為空、信心偏低或文字重複的短段，才擴大搜尋重辨一次。")
elif mode == "快速" and engine_name == "faster-whisper":
    st.caption("快速模式採用兩段批次辨識，保留原本搜尋範圍。批次處理會增加瞬間記憶體用量。")
if mode == "極速":
    st.warning("極速模式會縮小搜尋範圍，專有名詞與語句較容易誤辨，請對照錄音校正。")
terms = st.text_input("專有名詞提示（選填）", disabled=active)
st.caption("專有名詞會作為辨識熱詞提示，例如 dopamine、多巴胺、前額葉。請只填入錄音相關詞彙，避免錯誤提示影響辨識。")
initial_parallel = st.checkbox("新工作啟用切分兩段並行轉錄", value=False, disabled=active)
initial_word_timestamps = st.checkbox(
    "新工作啟用詞級時間戳（方便精確配對說話者；推論會較慢）",
    value=True, disabled=active or engine_name == "whisper.cpp")
speaker_timing = st.selectbox(
    "說話者辨識時機",
    ["手動辨識", "與轉錄同時開始", "每次暫停時開始", "轉錄完成後開始"],
    disabled=active,
    help="同時開始會占用額外 CPU 與記憶體，可能讓轉錄變慢；暫停時開始則會在第一次按下暫停時啟動整段分析。",
)
if st.button("開始轉錄", type="primary", disabled=recording is None or active or not backend_ready):
    try:
        job = TranscriptionJob(recording, size,
                               {"中文": "zh", "英文": "en", "自動偵測": None}[language],
                               mode, terms, backend=backend, parallel_enabled=initial_parallel,
                               speaker_timing={"手動辨識": "manual", "與轉錄同時開始": "during",
                                               "每次暫停時開始": "pause", "轉錄完成後開始": "after"}[speaker_timing],
                               speaker_token=st.session_state.get("speaker_token_global", ""),
                               word_timestamps=initial_word_timestamps)
        register_job(job)
        st.session_state.job = job
        st.query_params["job"] = job.id
        job.start()
        st.rerun()
    except Exception as error:
        st.error(f"無法開始轉錄：{error}")


@st.fragment(run_every=1)
def job_panel():
    job = st.session_state.get("job")
    if job is None:
        return
    if getattr(job, "deleted", False):
        st.info("此保存工作已刪除。")
        return
    state = job_snapshot(job)
    if not hasattr(job, "content_revision"):
        st.info("此工作仍使用更新前的引擎。保存進度後重新啟動程式，再開啟工作即可套用新版。")
    statuses = {
        "running": "轉錄中", "pausing": "等待目前辨識完成後暫停",
        "paused": "已暫停", "stopping": "等待目前辨識完成後停止",
        "stopped": "已停止", "completed": "轉錄完成", "failed": "轉錄失敗",
        "interrupted": "工作中斷，可從保存點接續",
    }
    st.subheader(f"{state['name']} — {statuses[state['status']]}")
    left, right, transcription_time = st.columns(3)
    left.metric("已完成轉錄的音檔時長", timestamp(state["processed"]).split(",")[0])
    right.metric("已經過時間", timestamp(state["elapsed"]).split(",")[0])
    transcription_elapsed = state.get("transcription_elapsed")
    transcription_time.metric(
        "實際轉錄耗時",
        timestamp(transcription_elapsed).split(",")[0] if transcription_elapsed is not None else "尚無紀錄",
    )
    timing_options = {
        "manual": "手動辨識", "during": "與轉錄同時開始",
        "pause": "每次暫停時開始", "after": "轉錄完成後開始",
    }
    timing_key = f"saved_speaker_timing_{job.id}"
    if timing_key not in st.session_state:
        st.session_state[timing_key] = timing_options.get(state.get("speaker_timing", "manual"), "手動辨識")
    selected_timing = st.selectbox("此工作的說話者辨識時機", list(timing_options.values()), key=timing_key)
    selected_timing_code = {value: key for key, value in timing_options.items()}[selected_timing]
    if selected_timing_code != state.get("speaker_timing", "manual"):
        try:
            job.set_speaker_timing(selected_timing_code, st.session_state.get("speaker_token_global", ""))
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(f"無法更新此工作的辨識時機：{error}")
    st.caption("已保存工作也可在此更改；選擇會保存並套用於此工作。每次暫停時，只分析實際暫停時已轉錄且尚未標記的部分。")
    if transcription_elapsed is None:
        st.caption("此工作在更新前啟動，尚未記錄實際轉錄耗時；仍可暫停、繼續與下載逐字稿。")
    st.caption("音檔時長包含已略過的靜音；已經過時間包含下載與載入。實際暫停後所有時間停止累計，繼續轉錄後恢復。")
    st.caption("實際轉錄耗時從開始辨識音檔起算，暫停期間停止計時。Whisper.cpp 包含每段子程序內的模型載入時間；舊工作從更新後的接續起累計。")
    st.caption(state["phase"])
    if hasattr(job, "set_parallel_enabled"):
        parallel_key = f"parallel_enabled_{job.id}"
        if parallel_key not in st.session_state:
            st.session_state[parallel_key] = state.get("parallel_enabled", False)
        parallel_enabled = st.toggle("切分兩段並行轉錄", key=parallel_key)
        if parallel_enabled != state.get("parallel_enabled", False):
            try:
                job.set_parallel_enabled(parallel_enabled)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"設定保存失敗：{error}")
        st.caption("可隨時切換，從下一個兩分鐘片段生效；目前片段會照原設定完成。選擇會自動保存。並行不保證更快，預設關閉。")
        if (state.get("chunk_parallel_enabled") is not None
                and state["status"] not in TERMINAL
                and state["chunk_parallel_enabled"] != parallel_enabled):
            st.info("設定已變更，等待下一個兩分鐘片段套用。")
        word_key = f"word_timestamps_{job.id}"
        if word_key not in st.session_state:
            st.session_state[word_key] = state.get("word_timestamps", True)
        word_enabled = st.toggle("詞級時間戳與細緻說話者標記", key=word_key,
                                 disabled=state.get("backend", {}).get("engine") == "whisper.cpp")
        if word_enabled != state.get("word_timestamps", True):
            try:
                job.set_word_timestamps(word_enabled)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"設定保存失敗：{error}")
        st.caption("可隨時切換，從下一個兩分鐘片段生效；關閉後仍保留逐字稿時間戳，說話者標籤改按整句配對，不能在句中換人處切開。詞級時間戳會增加推論時間。Whisper.cpp 不支援此選項。")
        if (state.get("chunk_word_timestamps") is not None
                and state["status"] not in TERMINAL
                and state["chunk_word_timestamps"] != word_enabled):
            st.info("設定已變更，下一個兩分鐘片段開始套用。")
    else:
        st.caption("此工作使用更新前的程式；重新啟動並從保存點接續後，即可使用並行開關。")
    if state.get("preparing_model"):
        if state["status"] == "paused":
            st.info("轉錄已暫停；若正在載入模型，載入會在背景完成。可隨時按「繼續轉錄」，模型準備完成後會自動開始辨識。")
        else:
            st.info("正在準備音訊快取或模型，完成後會自動開始辨識。")
    pause, resume, stop = st.columns(3)
    if pause.button("暫停", disabled=state["status"] != "running"):
        job.pause()
        st.rerun()
    if resume.button("繼續轉錄", disabled=state["status"] not in {"paused", "pausing"}):
        job.resume()
        st.rerun()
    if stop.button("停止轉錄", disabled=state["status"] in TERMINAL or state["status"] == "stopping"):
        job.stop()
        st.rerun()
    st.caption("暫停或停止會在目前模型載入／辨識回傳後生效。暫停可接續，停止會結束本次工作並保留已完成文字。")
    if state["error"]:
        st.error(state["error"])
    st.subheader("說話者辨識（選用）")
    st.caption(f"設定時機：{timing_options.get(state.get('speaker_timing', 'manual'), '手動辨識')}。權杖使用上方共用設定。")
    if not importlib.util.find_spec("pyannote"):
        st.caption("首次使用前，請在程式資料夾執行：.venv\\Scripts\\python.exe -m pip install -r requirements-speakers.txt")
    st.caption(f"說話者進度：已處理 {state.get('speaker_done', 0)}／{state['row_count']} 個文字片段；每完成約 30 秒就更新一次逐字稿。")
    st.caption("每次從上次完成的位置接續；手動辨識只處理按下按鈕時的已轉錄範圍。同步模式會在模型檢查點隨轉錄暫停／繼續，模型載入與單次運算不能立即中斷；尚未完成的分析批次可能在繼續後重算。")
    st.caption("使用較長上下文與跨段聲音特徵比對；信心不足會標示未確認。說話者編號仍可能誤配，重要內容請校對。")
    if state.get("speaker_status") in {"running", "waiting", "paused"}:
        st.info(state.get("speaker_phase") or "說話者辨識進行中。")
    elif state.get("speaker_status") == "failed":
        st.error(state.get("speaker_error", "說話者辨識失敗"))
    elif state.get("speaker_status") == "completed":
        st.success("本次指定範圍已完成；有新文字後可繼續辨識。")
    if st.button("辨識說話者", disabled=(state.get("speaker_status") in {"running", "waiting", "paused"}
                                              or state["row_count"] <= state.get("speaker_done", 0))):
        try:
            speaker_token = st.session_state.get("speaker_token_global", "")
            if not speaker_token.strip():
                raise ValueError("請先在上方的共用權杖設定中輸入並記住 Hugging Face Read 權杖。")
            job.start_speaker_diarization(speaker_token)
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(f"無法開始：{error}")
    if state["status"] in {"interrupted", "failed", "stopped"}:
        st.caption("接續時會重做最後未完成的兩分鐘分段，取代該段部分結果，避免重複文字。")
        if st.button("從保存點接續轉錄"):
            try:
                job.restart()
                st.rerun()
            except Exception as error:
                st.error(f"無法接續：{error}")
    if state["status"] == "completed":
        st.success(f"處理 {state['processed'] / 60:.1f} 分鐘錄音，總耗時 {state['elapsed'] / 60:.1f} 分鐘（不含暫停時間）。")
    if state["row_count"]:
        if hasattr(job, "export_payload"):
            output = job.export_payload()
        else:
            labels = state.get("speaker_labels") or []
            rows = [(a, b, f"{labels[i] or '未確認'}：{text}" if i < len(labels) else text)
                    for i, (a, b, text) in enumerate(state["rows"])]
            output = export_rows(rows)
            output['preview'] = export_rows(rows[-300:])['timed']
            output['downloads'] = {key: output[key].encode('utf-8-sig') for key in ('plain', 'timed', 'srt')}
        suffix = "_部分" if state["status"] != "completed" else ""
        st.caption("可隨時下載目前已完成內容，也可先暫停、下載，再繼續轉錄。")
        for label, field, extension, mime in [
            ("下載逐字稿 TXT", "plain", ".txt", "text/plain"),
            ("下載含時間逐字稿", "timed", "_時間.txt", "text/plain"),
            ("下載字幕 SRT", "srt", ".srt", "application/x-subrip"),
        ]:
            st.download_button(label, output['downloads'][field],
                               state["name"] + suffix + extension, mime=mime, on_click="ignore")
        st.caption("預覽顯示最近 300 段，下載檔案包含完整內容。")
        st.text_area("逐字稿預覽", output["preview"], height=350, disabled=True)
    elif state["status"] == "completed":
        st.warning("沒有辨識到語音。")
    if state["status"] in TERMINAL:
        if st.button("準備下一次轉錄"):
            del st.session_state.job
            st.query_params.clear()
            st.rerun()


job_panel()
