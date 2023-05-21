import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { logWithTimestamp} from './logWithTimestamp.js';

export async function RunGitHubRepoLoader() {
  logWithTimestamp ('start GitHubLoader');

  /* Initialize the LLM to use to answer the question */
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0.9,
  });

  // Load repo into docs
  const loader = new GithubRepoLoader(
    "https://github.com/hwchase17/langchainjs",
    { branch: "main", recursive: true, unknown: "error",
    accessToken: process.env.GITHUB_ACCESS_TOKEN }
  );
  const repoDocs = await loader.load();
  logWithTimestamp (`finished loading, repoDocs.length: ${repoDocs.length}`);

  /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.splitDocuments(repoDocs);
  logWithTimestamp (`finished splitting, docs.length: ${docs.length}`);

  /* Create the vectorstore */
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  logWithTimestamp ('finished creating vector store');

  // Save the vector store to a directory
  const directory = "../data/vectorStores/GitHubLoader";
  await vectorStore.save(directory);
  logWithTimestamp ('finished saving vector store');
  
  /* Create the chain */
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever()
  );

  /* Ask it a question */
  const question = "I want to load code files from a local code repository and ask you questions about that code.  I am open to just using the basic DirectoryLoader functionality.  Can you show me how to do that?";
  const res = await chain.call({ question, chat_history: [] });
  logWithTimestamp (res);

  /* Ask it a follow up question */
  const chatHistory = question + res.text;
  const followUpRes = await chain.call({
    question: "Please refine the answer and write concise code showing how to do this.",
    chat_history: chatHistory,
  });
  logWithTimestamp (followUpRes);
}
