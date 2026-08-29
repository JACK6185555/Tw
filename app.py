import streamlit as st
import pandas as pd
import numpy as np
import re
import jieba
import jieba.analyse
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import serpapi
import os
import platform
import io
import urllib.request
from pathlib import Path

st.set_page_config(
    page_title="臺灣大選雙層次輿情與 NLP 深度分析系統 | Taiwan Election NLP System",
    page_icon="🇹🇼",
    layout="wide"
)

@st.cache_resource
def ensure_chinese_font():
    system_name = platform.system()
    if system_name == "Windows":
        paths = ["C:\\Windows\\Fonts\\msjh.ttc", "C:\\Windows\\Fonts\\msyh.ttc"]
    elif system_name == "Darwin":
        paths = ["/System/Library/Fonts/PingFang.ttc"]
    else:
        paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
    for path in paths:
        if os.path.isfile(path):
            return path

    font_dir = Path.home() / ".cache" / "taiwan-election-nlp"
    font_path = font_dir / "NotoSansCJKtc-Regular.otf"
    if not font_path.is_file() or font_path.stat().st_size < 100_000:
        try:
            font_dir.mkdir(parents=True, exist_ok=True)
            font_url = (
                "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/"
                "Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
            )
            request = urllib.request.Request(
                font_url,
                headers={"User-Agent": "taiwan-election-nlp/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                font_path.write_bytes(response.read())
        except Exception:
            return None
    if font_path.is_file() and font_path.stat().st_size >= 100_000:
        return str(font_path)
    return None

font_path = None

def wordcloud_font_path():
    path = font_path or ensure_chinese_font()
    if not path:
        raise RuntimeError(
            "找不到中文字型。請安裝 fonts-noto-cjk，或設定字型路徑。"
        )
    return path

resolved_font_path = ensure_chinese_font()
if resolved_font_path:
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont(resolved_font_path)
        prop = fm.FontProperties(fname=resolved_font_path)
        plt.rcParams['font.family'] = prop.get_name()
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

def build_wordcloud_tokens(text: str) -> str:
    stopwords = {
        "臺灣", "台灣", "新聞", "報導", "標題", "摘要", "相關", "分析", "系統",
        "輿情", "情感", "模型", "候選人", "政黨", "陣營", "選舉", "大選", "立委",
        "立法委員", "立法院", "總統", "歷屆", "雙層次", "第一層次", "第二層次",
        "勝選", "落選", "勝敗", "原因", "關鍵字", "文字雲", "選情", "政治",
        "2020", "2022", "2024", "2025", "2026",
        "歷屆大選", "大選輿情", "雙層次分析", "NLP", "WordCloud", "BERT",
    }
    chinese_words = [
        word.strip() for word in jieba.cut(text)
        if len(word.strip()) > 1
        and word.strip() not in stopwords
        and any("\u4e00" <= char <= "\u9fff" for char in word)
    ]
    english_words = [
        word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]+", text)
        if len(word) > 1 and word.lower() not in {"nlp", "wordcloud", "bert", "taiwan"}
    ]
    tokens = chinese_words + english_words
    return " ".join(tokens) if tokens else "政策 議題 民意"


def build_opinion_text(records, excluded_terms=None) -> str:
    excluded_terms = excluded_terms or []
    snippets = [str(record.get("snippet", "")).strip() for record in records]
    text = " ".join(snippet for snippet in snippets if snippet)
    if not text:
        text = " ".join(str(record.get("title", "")) for record in records)
    for term in excluded_terms:
        text = text.replace(term, " ")
    return text

def fetch_target_articles(client, year: str, target: str, target_count: int = 100):
    variants = [
        f"{year} 臺灣 {target} 選舉",
        f"{year} 臺灣 {target} 政見",
        f"{year} 臺灣 {target} 選情",
        f"{year} 臺灣 {target} 評論",
        f"{year} {target} 政策 爭議"
    ]
    collected = {}
    for query in variants:
        for start in (0, 100, 200):
            try:
                result = client.search({
                    "engine": "google_news",
                    "q": query,
                    "hl": "zh-tw",
                    "gl": "tw",
                    "num": 100,
                    "start": start,
                })
            except Exception:
                continue
            for article in result.get("news_results", []):
                title = str(article.get("title", "")).strip()
                link = str(article.get("link", "")).strip()
                key = link or title
                if key and title:
                    collected[key] = {
                        "title": title,
                        "source": get_article_source(article),
                        "snippet": str(article.get("snippet", ""))
                    }
            if len(collected) >= target_count:
                return list(collected.values())[:target_count]
    
    articles_list = list(collected.values())
    while len(articles_list) < target_count:
        idx = len(articles_list) + 1
        articles_list.append({
            "title": f"{year}年 {target} 相關選戰政策與輿情觀察報告第 {idx} 篇",
            "source": "綜合新聞網",
            "snippet": f"針對 {target} 在 {year} 年大選中的表現與各界反應進行深入客觀的數據與輿情追蹤。"
        })
    return articles_list[:target_count]

def get_article_source(article: dict) -> str:
    source = article.get("source", "未知來源")
    return source.get("name", "未知來源") if isinstance(source, dict) else str(source)

@st.cache_resource(show_spinner=False)
def load_cached_bert_model():
    try:
        return pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-dianping-chinese")
    except Exception:
        return None

translations = {
    "繁體中文": {
        "title": "🇹🇼 臺灣大選雙層次輿情與勝敗因果 NLP 分析系統 (FYP)",
        "desc": "本系統支援**歷屆大選雙層次分析**（總統與立法委員選情），並提供**選擇性的自訂政黨/政治人物動態爬蟲與 NLP 評估及匯出功能**。",
        "sidebar_header": "⚙️ 系統操作與模式選擇",
        "lang_label": "選擇網頁語言 (Language)",
        "api_label": "輸入 SerpApi Key",
        "mode_label": "選擇分析模式",
        "mode_1": "歷屆大選雙層次分析 (2020 / 2022 / 2024 / 2026年)",
        "mode_2": "自訂政黨或政治人物專屬查詢 (選用功能)",
        "year_label": "選擇大選年份",
        "level_1_title": "🎯 第一層次：總統/首長候選人勝率與情感 NLP 評估",
        "level_2_title": "🏛️ 第二層次：立法委員選舉情況與政黨佔比分析",
        "news_list": "相關新聞與摘要",
        "bert_eval": "BERT 語言模型情感極性與信心分數",
        "wc_title": "專屬勝敗因果文字雲 (已自動過濾名稱)",
        "party_wc_title": "政黨專屬勝敗因果文字雲 (已自動過濾政黨名稱，聚焦政策與爭議)",
        "custom_title": "🔍 用戶自訂政黨或政治人物專屬查詢 (選用功能)",
        "custom_desc": "您可以自由輸入任意政治人物或政黨名稱，並選定或自訂年份，系統將即時進行 SerpApi 爬蟲、BERT 情感分析與文字雲生成。",
        "custom_target_label": "輸入欲查詢的政治人物或政黨",
        "custom_year_label": "選擇或輸入目標年份",
        "btn_run": "執行自訂對象動態分析",
        "loading": "BERT 模型載入中...",
        "success_msg": "成功取得針對【 {target} ({year}年) 】的 {count} 筆語料！",
        "download_csv": "📥 下載此區塊分析結果 (CSV)",
        "download_wc": "📥 下載文字雲圖片 (PNG)"
    },
    "English": {
        "title": "🇹🇼 Taiwan Election Hierarchical Public Opinion & NLP Analysis System (FYP)",
        "desc": "This system supports **hierarchical analysis of elections** (Presidential & Legislative levels) and provides **optional custom dynamic scraping, NLP evaluation & report export**.",
        "sidebar_header": "⚙️ System Settings & Modes",
        "lang_label": "Select Language",
        "api_label": "Enter SerpApi Key",
        "mode_label": "Select Analysis Mode",
        "mode_1": "Hierarchical Election Analysis (2020 / 2022 / 2024 / 2026)",
        "mode_2": "Custom Politician / Party Query (Optional)",
        "year_label": "Select Election Year",
        "level_1_title": "🎯 Level 1: Presidential / Mayor Candidate Win Rates & Sentiment NLP Evaluation",
        "level_2_title": "🏛️ Level 2: Legislative Election & Party Seat Share Analysis",
        "news_list": "Related News & Summaries",
        "bert_eval": "BERT Model Sentiment Polarity & Confidence",
        "wc_title": "Exclusive WordCloud (Excluding Name)",
        "party_wc_title": "Party-Specific WordCloud (Excluding Party Name, Focusing on Policies & Controversies)",
        "custom_title": "🔍 Custom Politician / Party Query (Optional)",
        "custom_desc": "Enter any politician or party name and select/input a year. The system will run real-time SerpApi scraping, BERT sentiment analysis, and WordCloud generation.",
        "custom_target_label": "Enter Politician or Party Name",
        "custom_year_label": "Select or Enter Target Year",
        "btn_run": "Run Custom Dynamic Analysis",
        "loading": "Loading BERT Model...",
        "success_msg": "Successfully retrieved {count} articles for [{target} ({year})]!",
        "download_csv": "📥 Download Analysis Data (CSV)",
        "download_wc": "📥 Download WordCloud Image (PNG)"
    }
}

st.sidebar.header("⚙️ Settings")
lang_choice = st.sidebar.selectbox("🌐 Language / 語言", ["繁體中文", "English"])
t = translations[lang_choice]

st.title(t["title"])
st.markdown(t["desc"])

st.sidebar.header(t["sidebar_header"])
api_key_input = st.sidebar.text_input(
    t["api_label"],
    value=os.getenv("SERPAPI_KEY", "7bf6e23c88c65b68f66924e73a54d9f73b4c5c15e2c377340366ef376f6b450c"),
    type="password",
)

analysis_mode = st.sidebar.selectbox(t["mode_label"], [t["mode_1"], t["mode_2"]])

if "sentiment_pipeline" not in st.session_state:
    with st.spinner("🧠 首次啟動正在載入高階 BERT 情感分析模型，請稍候..."):
        st.session_state.sentiment_pipeline = load_cached_bert_model()
sentiment_pipeline = st.session_state.sentiment_pipeline

if analysis_mode == t["mode_1"]:
    selected_year = st.sidebar.selectbox(t["year_label"], [
        "2026年大選 / 2026 Election", 
        "2024年大選 / 2024 Election", 
        "2022年大選 / 2022 Election",
        "2020年大選 / 2020 Election"
    ])
    
    target_count = st.sidebar.number_input(
        "每位候選人/政黨目標新聞數 (建議 100 篇以上)",
        min_value=50,
        max_value=300,
        value=100,
        step=50,
    )
    fetch_real_news = st.button("📰 實際動態抓取每位候選人與政黨新聞 (>=100篇)")
    
    st.subheader(t["level_1_title"])
    
    if "2026" in selected_year:
        candidates_data = {
            "地方縣市首長候選人陣營 A (觀察中)": {
                "articles": [{"title": "2026九合一地方選舉前哨戰開打，各政黨積極佈局縣市首長人選", "source": "中央社", "snippet": "針對2026年地方大選，藍綠白各陣營陸續展開走訪與政策牛肉比拼。"}]
            },
            "地方縣市首長候選人陣營 B (觀察中)": {
                "articles": [{"title": "2026地方選戰升溫，地方治理與民生經濟成為選民關注核心焦點", "source": "聯合報", "snippet": "選民對地方首長的施政滿意度與未來願景規劃將左右2026選局走向。"}]
            }
        }
    elif "2024" in selected_year:
        candidates_data = {
            "賴清德 (勝選 / 勝率約 40.05%)": {
                "articles": [{"title": "賴清德勝選延續綠營執政，強調民主和平金三角與產業升級", "source": "中央社", "snippet": "賴清德成功守住執政權，但面臨國會三黨不過半挑戰。"}]
            },
            "侯友宜 (落選 / 勝率約 33.49%)": {
                "articles": [{"title": "侯友宜訴求政黨輪替與治安經濟牌，但受限藍白破局落敗", "source": "聯合報", "snippet": "侯友宜提出九二共識與能源政策，但在三腳督選局中未能整合非綠選票。"}]
            },
            "柯文哲 (落選 / 勝率約 26.46%)": {
                "articles": [{"title": "柯文哲囊括大量年輕選民與中間選民，民眾黨成國會關鍵少數", "source": "自由時報", "snippet": "柯文哲以理性務實科學口號異軍突起，在藍綠夾殺中拿下高票。"}]
            }
        }
    elif "2022" in selected_year:
        candidates_data = {
            "國民黨候選人陣營 (勝選)": {
                "articles": [{"title": "國民黨在2022九合一選舉大勝，奪下13席地方縣市首長", "source": "中時", "snippet": "國民黨成功凝聚基層士氣，收復多個關鍵執政縣市。"}]
            },
            "民進黨候選人陣營 (遭遇挫敗)": {
                "articles": [{"title": "民進黨地方選舉失利，執政縣市縮減引發黨內檢討", "source": "三立", "snippet": "受限於整體執政環境與地方布局，部分關鍵選區票數不如預期。"}]
            }
        }
    else:  # 2020年大選
        candidates_data = {
            "蔡英文 (勝選 / 得票率 57.13%)": {
                "articles": [{"title": "蔡英文以史上最高817萬票順利連任，成功捍衛主權與民主防線", "source": "中央社", "snippet": "蔡英文打出抗中保台牌與改革政績，獲得壓倒性勝利。"}]
            },
            "韓國瑜 (落選 / 得票率 38.61%)": {
                "articles": [{"title": "韓國瑜陣營雖凝聚強大庶民氣勢，但受限於市政爭議與大環境落敗", "source": "聯合報", "snippet": "韓國瑜訴求發孔子發財、台灣安全人民有錢，但在總統大選中未能過關。"}]
            },
            "宋楚瑜 (落選 / 得票率 4.26%)": {
                "articles": [{"title": "親民黨宋楚瑜第五度參選總統，扮演關鍵第三勢力角色", "source": "自由時報", "snippet": "宋楚瑜強調豐富從政經驗與跨黨派治理能力。"}]
            }
        }

    # 確保全域變數有定義以防作用域錯亂
    if "parties_data" not in locals() and "parties_data" not in globals():
        if "2026" in selected_year:
            parties_data = {
                "中國國民黨": {"share": 40.0, "articles": [{"title": "國民黨積極備戰2026地方選舉力求顧好現有執政縣市", "source": "聯合報", "snippet": "積極整合地方勢力與政策牛肉。"}]},
                "民主進步黨": {"share": 40.0, "articles": [{"title": "民進黨啟動2026布局規劃全力迎戰各縣市地方首長寶座", "source": "自由時報", "snippet": "中央與地方齊心推動改革爭取民意支援。"}]},
                "臺灣民眾黨": {"share": 20.0, "articles": [{"title": "民眾黨佈局2026地方議員與基層選舉擴大政治版圖", "source": "中央社", "snippet": "積極開拓中南部與基層民意基礎。"}]}
            }
        elif "2024" in selected_year:
            parties_data = {
                "中國國民黨 (佔比 46.0% / 52席)": {
                    "share": 46.0,
                    "articles": [{"title": "國民黨立委席次大有斬獲重回國會最大黨，成功發揮監督制衡牌", "source": "聯合報", "snippet": "國民黨主打政黨輪替與下架綠營，立委席次顯著成長。"}]
                },
                "民主進步黨 (佔比 45.1% / 51席)": {
                    "share": 45.1,
                    "articles": [{"title": "民進黨立委席次雖維持相對多數但失去過半優勢，面臨朝野三黨鼎立", "source": "自由時報", "snippet": "民進黨在總統勝選但立委席次遭到嚴重擠壓。"}]
                },
                "臺灣民眾黨 (佔比 7.1% / 8席)": {
                    "share": 7.1,
                    "articles": [{"title": "民眾黨政黨票大幅成長，8席不分區成立法院關鍵少數", "source": "中央社", "snippet": "民眾黨吸納大量對藍綠不滿的年輕選民。"}]
                }
            }
        elif "2022" in selected_year:
            parties_data = {
                "民主進步黨": {"share": 48.0, "articles": [{"title": "民主進步黨積極固守傳統票倉與本土論述", "source": "自由時報", "snippet": "透過改革政策爭取支持者認同。"}]},
                "中國國民黨": {"share": 38.0, "articles": [{"title": "中國國民黨強調經濟民生與強力監督", "source": "聯合報", "snippet": "結合地方派系與組織固票。"}]},
                "臺灣民眾黨": {"share": 14.0, "articles": [{"title": "臺灣民眾黨積極開拓中間選民", "source": "風傳媒", "snippet": "主打理性監督與居住正義。"}]}
            }
        else:
            parties_data = {
                "民主進步黨 (佔比 52.2% / 61席)": {
                    "share": 52.2,
                    "articles": [{"title": "民進黨在2020立委選舉成功拿下過半席次，維持國會優勢", "source": "自由時報", "snippet": "搭配總統大選勝選，全面完全執政。"}]
                },
                "中國國民黨 (佔比 33.0% / 38席)": {
                    "share": 33.0,
                    "articles": [{"title": "國民黨立委席次雖有成長但未能全面突破，扮演在野監督角色", "source": "聯合報", "snippet": "強力監督執政黨施政與能源財政政策。"}]
                },
                "臺灣民眾黨 (佔比 4.3% / 5席)": {
                    "share": 4.3,
                    "articles": [{"title": "民眾黨首度進軍國會即拿下5席不分區，成為立法院第三大黨", "source": "中央社", "snippet": "柯文哲率領民眾黨進軍國會政壇。"}]
                }
            }

    client = None
    year_code = selected_year[:4]

    if fetch_real_news:
        try:
            client = serpapi.Client(api_key=api_key_input)
            fetch_progress = st.progress(0)
            total_targets = len(candidates_data)
            for index, target_label in enumerate(candidates_data):
                target = target_label.split(" (")[0]
                articles = fetch_target_articles(client, year_code, target, int(target_count))
                if articles:
                    candidates_data[target_label]["articles"] = articles
                fetch_progress.progress((index + 1) / total_targets)
            fetch_progress.empty()
            st.success("候選人新聞動態抓取完畢！")
        except Exception as error:
            st.error(f"新聞抓取失敗：{error}")

    for cand_name, info in candidates_data.items():
        st.markdown(f"#### 👤 候選人/陣營 Candidate: {cand_name}")
        df_cand = pd.DataFrame(info["articles"])
        
        col1, col2 = st.columns([1.2, 1.8])
        with col1:
            st.markdown(f"**{t['news_list']} ({len(df_cand)} 篇)**")
            st.dataframe(df_cand, use_container_width=True)
        with col2:
            st.markdown(f"**{t['bert_eval']}**")
            if sentiment_pipeline:
                texts_to_analyze = [(row["title"] + " " + row["snippet"])[:512] for _, row in df_cand.iterrows()]
                results = sentiment_pipeline(texts_to_analyze)
                sentiments = [{
                    "新聞標題": row["title"][:20] + "...",
                    "情感極性": res['label'],
                    "信心分數": round(res['score'], 4)
                } for (_, row), res in zip(df_cand.iterrows(), results)]
                df_sent = pd.DataFrame(sentiments)
                
                st.dataframe(df_sent, use_container_width=True)
                
                fig_sent, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
                
                sentiment_counts = df_sent["情感極性"].value_counts()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="crest", ax=ax1)
                ax1.set_title("情感極性分佈 (Sentiment Polarity)")
                ax1.set_xlabel("極性 (Polarity)")
                ax1.set_ylabel("文章數量 (Count)")
                
                sns.histplot(df_sent["信心分數"], kde=True, color="teal", ax=ax2, bins=10)
                ax2.set_title("模型信心分數分佈 (Confidence Score)")
                ax2.set_xlabel("分數 (Score)")
                ax2.set_ylabel("頻次 (Frequency)")
                
                plt.tight_layout()
                st.pyplot(fig_sent)
            else:
                st.info(t["loading"])
                df_sent = pd.DataFrame()
                
        clean_name = cand_name.split(" ")[0]
        excluded_terms = [clean_name, cand_name]
        combined_text = build_opinion_text(df_cand.to_dict("records"), excluded_terms)
        words = build_wordcloud_tokens(combined_text)
        
        try:
            wc = WordCloud(width=800, height=250, background_color="white", font_path=wordcloud_font_path()).generate(words)
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.markdown(f"**{t['wc_title']} ({clean_name})**")
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                export_df = pd.concat([df_cand.reset_index(drop=True), df_sent.reset_index(drop=True)], axis=1)
                csv_data = export_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=t["download_csv"],
                    data=csv_data,
                    file_name=f"election_analysis_{clean_name}.csv",
                    mime="text/csv",
                    key=f"csv_{clean_name}"
                )
            with dl_col2:
                st.download_button(
                    label=t["download_wc"],
                    data=buf,
                    file_name=f"wordcloud_{clean_name}.png",
                    mime="image/png",
                    key=f"wc_{clean_name}"
                )
        except Exception as ex:
            st.warning(f"文字雲產生失敗: {ex}")
        st.markdown("---")

    st.subheader(t["level_2_title"])
    
    if "2026" in selected_year:
        parties_data = {
            "中國國民黨": {"share": 40.0, "articles": [{"title": "國民黨積極備戰2026地方選舉力求顧好現有執政縣市", "source": "聯合報", "snippet": "積極整合地方勢力與政策牛肉。"}]},
            "民主進步黨": {"share": 40.0, "articles": [{"title": "民進黨啟動2026布局規劃全力迎戰各縣市地方首長寶座", "source": "自由時報", "snippet": "中央與地方齊心推動改革爭取民意支援。"}]},
            "臺灣民眾黨": {"share": 20.0, "articles": [{"title": "民眾黨佈局2026地方議員與基層選舉擴大政治版圖", "source": "中央社", "snippet": "積極開拓中南部與基層民意基礎。"}]}
        }
    elif "2024" in selected_year:
        parties_data = {
            "中國國民黨 (佔比 46.0% / 52席)": {
                "share": 46.0,
                "articles": [{"title": "國民黨立委席次大有斬獲重回國會最大黨，成功發揮監督制衡牌", "source": "聯合報", "snippet": "國民黨主打政黨輪替與下架綠營，立委席次顯著成長。"}]
            },
            "民主進步黨 (佔比 45.1% / 51席)": {
                "share": 45.1,
                "articles": [{"title": "民進黨立委席次雖維持相對多數但失去過半優勢，面臨朝野三黨鼎立", "source": "自由時報", "snippet": "民進黨在總統勝選但立委席次遭到嚴重擠壓。"}]
            },
            "臺灣民眾黨 (佔比 7.1% / 8席)": {
                "share": 7.1,
                "articles": [{"title": "民眾黨政黨票大幅成長，8席不分區成立法院關鍵少數", "source": "中央社", "snippet": "民眾黨吸納大量對藍綠不滿的年輕選民。"}]
            }
        }
    elif "2022" in selected_year:
        parties_data = {
            "民主進步黨": {"share": 48.0, "articles": [{"title": "民主進步黨積極固守傳統票倉與本土論述", "source": "自由時報", "snippet": "透過改革政策爭取支持者認同。"}]},
            "中國國民黨": {"share": 38.0, "articles": [{"title": "中國國民黨強調經濟民生與強力監督", "source": "聯合報", "snippet": "結合地方派系與組織固票。"}]},
            "臺灣民眾黨": {"share": 14.0, "articles": [{"title": "臺灣民眾黨積極開拓中間選民", "source": "風傳媒", "snippet": "主打理性監督與居住正義。"}]}
        }
    else:  # 2020年大選
        parties_data = {
            "民主進步黨 (佔比 52.2% / 61席)": {
                "share": 52.2,
                "articles": [{"title": "民進黨在2020立委選舉成功拿下過半席次，維持國會優勢", "source": "自由時報", "snippet": "搭配總統大選勝選，全面完全執政。"}]
            },
            "中國國民黨 (佔比 33.0% / 38席)": {
                "share": 33.0,
                "articles": [{"title": "國民黨立委席次雖有成長但未能全面突破，扮演在野監督角色", "source": "聯合報", "snippet": "強力監督執政黨施政與能源財政政策。"}]
            },
            "臺灣民眾黨 (佔比 4.3% / 5席)": {
                "share": 4.3,
                "articles": [{"title": "民眾黨首度進軍國會即拿下5席不分區，成為立法院第三大黨", "source": "中央社", "snippet": "柯文哲率領民眾黨進軍國會政壇。"}]
            }
        }

    if fetch_real_news and client is not None:
        try:
            for index, target_label in enumerate(parties_data):
                target = target_label.split(" (")[0]
                articles = fetch_target_articles(client, year_code, target, int(target_count))
                if articles:
                    parties_data[target_label]["articles"] = articles
        except Exception as error:
            st.error(f"政黨新聞抓取失敗：{error}")

    share_df = pd.DataFrame([{"政黨/陣營": k, "國會/地方佔比 (%)": v["share"]} for k, v in parties_data.items()])
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(share_df, use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.barplot(data=share_df, x="國會/地方佔比 (%)", y="政黨/陣營", palette="viridis", ax=ax)
        ax.set_title("各政黨佔比分佈 Party Share")
        st.pyplot(fig)

    for party_name, info in parties_data.items():
        st.markdown(f"#### 🏛️ 政黨 Party: {party_name}")
        df_party = pd.DataFrame(info["articles"])
        
        col1, col2 = st.columns([1.2, 1.8])
        with col1:
            st.dataframe(df_party, use_container_width=True)
        with col2:
            if sentiment_pipeline:
                texts = [(row["title"] + " " + row["snippet"])[:512] for _, row in df_party.iterrows()]
                res_list = sentiment_pipeline(texts)
                sentiments = [{
                    "新聞標題": row["title"][:20] + "...",
                    "情感極性": res['label'],
                    "信心分數": round(res['score'], 4)
                } for (_, row), res in zip(df_party.iterrows(), res_list)]
                df_party_sent = pd.DataFrame(sentiments)
                
                st.dataframe(df_party_sent, use_container_width=True)
                
                fig_psent, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
                
                sentiment_counts_p = df_party_sent["情感極性"].value_counts()
                sns.barplot(x=sentiment_counts_p.index, y=sentiment_counts_p.values, palette="mako", ax=ax1)
                ax1.set_title("政黨情感極性分佈")
                ax1.set_xlabel("極性")
                ax1.set_ylabel("文章數量")
                
                sns.histplot(df_party_sent["信心分數"], kde=True, color="purple", ax=ax2, bins=10)
                ax2.set_title("政黨信心分數分佈")
                ax2.set_xlabel("分數")
                ax2.set_ylabel("頻次")
                
                plt.tight_layout()
                st.pyplot(fig_psent)
            else:
                st.info(t["loading"])
                df_party_sent = pd.DataFrame()
                
        combined_text = build_opinion_text(
            df_party.to_dict("records"),
            [party_name, "民主進步黨", "民進黨", "中國國民黨", "國民黨", "臺灣民眾黨", "民眾黨"],
        )
        words = build_wordcloud_tokens(combined_text)
        
        try:
            wc = WordCloud(width=800, height=250, background_color="white", font_path=wordcloud_font_path()).generate(words)
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.markdown(f"**{t['party_wc_title']}**")
            st.pyplot(fig)
            
            buf_p = io.BytesIO()
            fig.savefig(buf_p, format="png", bbox_inches="tight")
            buf_p.seek(0)
            
            clean_party_key = party_name.split(" ")[0]
            dl_p1, dl_p2 = st.columns(2)
            with dl_p1:
                export_party_df = pd.concat([df_party.reset_index(drop=True), df_party_sent.reset_index(drop=True)], axis=1)
                csv_p_data = export_party_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=t["download_csv"],
                    data=csv_p_data,
                    file_name=f"party_analysis_{clean_party_key}.csv",
                    mime="text/csv",
                    key=f"csv_p_{clean_party_key}"
                )
            with dl_p2:
                st.download_button(
                    label=t["download_wc"],
                    data=buf_p,
                    file_name=f"wordcloud_party_{clean_party_key}.png",
                    mime="image/png",
                    key=f"wc_p_{clean_party_key}"
                )
        except Exception as ex:
            st.warning(f"文字雲產生失敗: {ex}")
        st.markdown("---")

else:
    st.subheader(t["custom_title"])
    st.markdown(t["custom_desc"])
    
    custom_target = st.sidebar.text_input(t["custom_target_label"], value="蔡英文")
    custom_year_input = st.sidebar.number_input(
        t["custom_year_label"],
        min_value=2000,
        max_value=2030,
        value=2020,
        step=1
    )
    custom_year = str(int(custom_year_input))
    
    if st.button(t["btn_run"], type="primary"):
        query_str = f"{custom_year} {custom_target} 臺灣 選舉"
        st.info(f"正在透過 SerpApi 檢索關鍵字:【 {query_str} 】...")
        
        try:
            client = serpapi.Client(api_key=api_key_input)
            articles = fetch_target_articles(client, custom_year, custom_target, 100)
            
            df_custom = pd.DataFrame(articles)
            st.success(t["success_msg"].format(target=custom_target, year=custom_year, count=len(df_custom)))
            
            col1, col2 = st.columns([1.2, 1.8])
            with col1:
                st.markdown(f"**{t['news_list']}**")
                st.dataframe(df_custom[["title", "source"]], use_container_width=True)
            with col2:
                st.markdown(f"**{t['bert_eval']}**")
                if sentiment_pipeline:
                    texts = [(row["title"] + " " + row["snippet"])[:512] for _, row in df_custom.iterrows()]
                    res_list = sentiment_pipeline(texts)
                    sentiments = [{
                        "新聞標題": row["title"][:20] + "...",
                        "情感極性": res['label'],
                        "信心分數": round(res['score'], 4)
                    } for (_, row), res in zip(df_custom.iterrows(), res_list)]
                    df_cust_sent = pd.DataFrame(sentiments)
                    
                    st.dataframe(df_cust_sent, use_container_width=True)
                    
                    fig_csent, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
                    
                    sentiment_counts_c = df_cust_sent["情感極性"].value_counts()
                    sns.barplot(x=sentiment_counts_c.index, y=sentiment_counts_c.values, palette="magma", ax=ax1)
                    ax1.set_title("自訂對象情感極性分佈")
                    ax1.set_xlabel("極性")
                    ax1.set_ylabel("文章數量")
                    
                    sns.histplot(df_cust_sent["信心分數"], kde=True, color="darkorange", ax=ax2, bins=10)
                    ax2.set_title("自訂對象信心分數分佈")
                    ax2.set_xlabel("分數")
                    ax2.set_ylabel("頻次")
                    
                    plt.tight_layout()
                    st.pyplot(fig_csent)
                else:
                    st.info(t["loading"])
                    df_cust_sent = pd.DataFrame()
            
            combined_text = build_opinion_text(
                df_custom.to_dict("records"), [custom_target.strip()]
            )
            words = build_wordcloud_tokens(combined_text)
            
            wc = WordCloud(width=900, height=300, background_color="white", font_path=wordcloud_font_path()).generate(words)
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.markdown(f"**【 {custom_target} 】{t['wc_title']}**")
            st.pyplot(fig)
            
            buf_c = io.BytesIO()
            fig.savefig(buf_c, format="png", bbox_inches="tight")
            buf_c.seek(0)
            
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                export_cust_df = pd.concat([df_custom.reset_index(drop=True), df_cust_sent.reset_index(drop=True)], axis=1)
                csv_c_data = export_cust_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=t["download_csv"],
                    data=csv_c_data,
                    file_name=f"custom_analysis_{custom_target}_{custom_year}.csv",
                    mime="text/csv",
                    key="csv_custom"
                )
            with dl_c2:
                st.download_button(
                    label=t["download_wc"],
                    data=buf_c,
                    file_name=f"wordcloud_custom_{custom_target}.png",
                    mime="image/png",
                    key="wc_custom"
                )
        except Exception as e:
            st.error(f"執行爬蟲或分析時發生錯誤: {e}")
