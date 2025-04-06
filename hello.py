# LangChain + Faiss + Gemini による RAG システム（テキストファイル対応版）

# ステップ1: 必要なライブラリのインストール
# pip install langchain langchain-google-genai faiss-cpu python-dotenv

# ステップ2: 必要なライブラリのインポート
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory

# ステップ3: APIキーの設定
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set in .env file.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ステップ4: テキストファイルの読み込みと前処理
def load_and_process_documents(file_paths):
    # ドキュメントの読み込み
    documents = []
    for file_path in file_paths:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"ファイル読み込み成功: {file_path}")
        except Exception as e:
            print(f"ファイル読み込みエラー: {file_path} - {e}")

    if not documents:
        raise ValueError(
            "有効なドキュメントがありません。ファイルパスを確認してください。"
        )

    # ドキュメントをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print(f"ドキュメントを {len(chunks)} チャンクに分割しました")
    return chunks


# ステップ5: ベクトルデータベースの作成
def create_vector_db(chunks):
    # Geminiのエンベディングモデルを使用
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # FAISSベクトルストアの作成
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("ベクトルデータベースを作成しました")
    return vector_store


# ステップ6: RAGチェーンの構築
def build_rag_chain(vector_store):
    # Gemini LLMの初期化
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # 会話履歴を保持するためのメモリの作成
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # メモリの出力キーを明示的に指定
    )

    # 検索ベースの会話チェーンを作成
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",  # チェーンの出力キーを明示的に指定
    )

    return chain


# ステップ7: 質問応答機能の実装
def ask_question(chain, question):
    response = chain.invoke({"question": question})

    print("\n質問:", question)
    print("\n回答:", response["answer"])

    # ソースドキュメントの表示
    print("\n参照ドキュメント:")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\nソース {i + 1}:\n{doc.page_content[:200]}...\n")

    return response


# ベクトルストアの保存と読み込み機能
def save_vector_store(vector_store, file_path="faiss_index"):
    vector_store.save_local(file_path)
    print(f"ベクトルストアを {file_path} に保存しました")


def load_vector_store(file_path="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(file_path, embeddings)
    print(f"ベクトルストアを {file_path} から読み込みました")
    return vector_store


# サンプルファイルの作成
def create_sample_files():
    # ディレクトリの作成
    os.makedirs("root/data", exist_ok=True)

    # サンプルファイル1の作成
    with open("root/data/doc1.txt", "w", encoding="utf-8") as f:
        f.write("""# 人工知能（AI）の基礎
        
人工知能（Artificial Intelligence、AI）とは、人間の知能を模倣し、学習、問題解決、パターン認識などのタスクを実行するコンピュータシステムのことです。

## 機械学習（Machine Learning）

機械学習は、AIの一分野で、データから学習し、明示的にプログラムされることなく改善するアルゴリズムの開発に焦点を当てています。主な種類には以下があります：

1. **教師あり学習** - ラベル付きデータから学習
2. **教師なし学習** - パターンを自動的に見つける
3. **強化学習** - 行動と報酬から学習

## ディープラーニング（Deep Learning）

ディープラーニングは、機械学習のサブセットで、人間の脳の構造と機能を模倣したニューラルネットワークを使用します。これにより、画像認識、自然言語処理、音声認識などの複雑なタスクが可能になります。

## AIの応用分野

AIは様々な分野で応用されています：

- **ヘルスケア** - 疾病診断、薬物開発
- **金融** - 詐欺検出、アルゴリズム取引
- **自動運転** - 障害物検出、経路計画
- **自然言語処理** - 翻訳、感情分析、チャットボット""")

    # サンプルファイル2の作成
    with open("root/data/doc2.txt", "w", encoding="utf-8") as f:
        f.write("""# RAG（Retrieval-Augmented Generation）について

RAG（検索拡張生成）は、大規模言語モデル（LLM）の能力を外部知識で拡張するアプローチです。

## RAGの仕組み

1. **検索（Retrieval）**: ユーザークエリに関連する情報を外部知識ベースから検索します。
2. **拡張（Augmentation）**: 検索された情報をLLMの入力に追加します。
3. **生成（Generation）**: 拡張された入力に基づいて、LLMが回答を生成します。

## RAGの利点

- **最新情報へのアクセス**: LLMの訓練データ以降の情報を利用できます。
- **事実の正確性向上**: 外部ソースからの事実に基づいた回答が可能です。
- **透明性**: 情報源を引用できるため、回答の信頼性が向上します。
- **ドメイン特化**: 特定分野の専門知識をLLMに提供できます。

## RAGの実装方法

RAGを実装するための一般的なステップ：

1. **知識ベースの構築**: 関連文書を収集し、前処理します。
2. **埋め込み（Embedding）**: 文書をベクトル表現に変換します。
3. **ベクトルデータベースの作成**: 効率的な検索のために文書ベクトルを保存します。
4. **検索エンジンの構築**: 類似性検索を実装します。
5. **LLMとの統合**: 検索結果をLLMのプロンプトに組み込みます。

## RAGの課題

- **関連性のある文書の検索**: 質問に本当に関連する文書を見つけることが重要です。
- **文脈の維持**: 複数の文書からの情報を一貫した形で統合する必要があります。
- **計算コスト**: 検索と生成の両方のステップが必要なため、処理時間とリソースが増加します。""")

    print("サンプルファイルを作成しました：")
    print("- root/data/doc1.txt")
    print("- root/data/doc2.txt")

    return ["root/data/doc1.txt", "root/data/doc2.txt"]


# メイン実行部分
def main():
    # サンプルファイルの作成
    file_paths = create_sample_files()

    # ドキュメントの処理
    chunks = load_and_process_documents(file_paths)

    # ベクトルDBの作成
    vector_store = create_vector_db(chunks)

    # ベクトルストアの保存（オプション）
    save_vector_store(vector_store, "faiss_index")

    # RAGチェーンの構築
    chain = build_rag_chain(vector_store)

    # 対話ループ
    print("\nRAGシステムの準備ができました。質問をどうぞ（終了するには 'exit' と入力）")
    while True:
        question = input("\nあなたの質問: ")
        if question.lower() == "exit":
            break
        ask_question(chain, question)


if __name__ == "__main__":
    main()
