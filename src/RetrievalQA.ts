import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain, LLMChain, RetrievalQAChain, VectorDBQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { Document } from "langchain/document";
import { parseSitemap } from "./parseSitemap.ts";
import { logWithTimestamp} from './logWithTimestamp.js';
import { PromptTemplate } from "langchain";
import { ChainTool, SerpAPI, Tool } from "langchain/tools";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { PlanAndExecuteAgentExecutor } from "langchain/experimental/plan_and_execute";
import { ChatOpenAI } from "langchain/chat_models";

export const RetrievalQA = async () => {
  /* Initialize the LLM to use to answer the question */
  // const model = new OpenAI({
  //   modelName: "gpt-3.5-turbo",
  //   openAIApiKey: process.env.OPENAI_API_KEY,
  //   temperature: 0,
  // });
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });

  // get URLs from sitemap
  logWithTimestamp('parsing site map...');
  const urls = await parseSitemap('https://js.langchain.com/sitemap.xml');
  logWithTimestamp('finished parsing site map');

  // const urls = ["https://js.langchain.com/docs/modules/indexes/vector_stores/"];

  // load all the content into a docs array
  logWithTimestamp('loading all these pages with cheerio...');
  logWithTimestamp(urls);
  const documents: Document[] = [];
  for (const url of urls) {
    const loader = new CheerioWebBaseLoader(url);
    const documentsLoaded = await loader.load();
    documents.push(...documentsLoaded);
  }
  logWithTimestamp(`finished loading pages with cheerio, loaded ${documents.length} documents`);

  /* Split the text into chunks */
  logWithTimestamp('splitting the text into chunks...');
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.splitDocuments(documents);
  logWithTimestamp(`finished splitting text, split into ${docs.length} documents`);

  /* Create the vectorstore */
  logWithTimestamp('creating the vectorstore...');
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  logWithTimestamp('finished creating the vectorstore');

  // Save the vector store to a directory
  logWithTimestamp ('saving vector store');
  const directory = "../data/vectorStores/EntireLangchainJsDocs";
  await vectorStore.save(directory);
  logWithTimestamp ('finished saving vector store');
  
  // /* Create the chain */
  // const chain = ConversationalRetrievalQAChain.fromLLM(
  //   model,
  //   vectorStore.asRetriever()
  // );

  // /* Ask it a question */
  // const question = "If the context is not relevant, please answer the question by using your own knowledge about the topic.  What years was the chevrolet SS made?";
  // logWithTimestamp(`asking the first question ${question}`);
  // const res = await chain.call({ question, chat_history: [] });
  // console.log(res);

  // /* Ask it a follow up question */
  // const followUpQuestion = "Please the javascript code that could power this chatbot.";
  // logWithTimestamp(`asking the follow up question ${followUpQuestion}`);

  // const chatHistory = question + res.text;
  // const followUpRes = await chain.call({
  //   question: followUpQuestion,
  //   chat_history: chatHistory,
  // });
  // console.log(followUpRes);

  /*
  const prompt = `If the langchain documentation context is not relevant, 
  please answer the question by using your own knowledge about the topic. Even if the context
  is relevant, please incorporate your own knowledge with the context to give the most rich answer.
  
  {context}
  
  Question: {question}`;

  const promptTemplate = new PromptTemplate({template: prompt, inputVariables: ["context", "question"]});
*/

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
// const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {prompt: promptTemplate});
  // const chain = VectorDBQAChain.fromLLM(model, vectorStore);

  const qaTool = new ChainTool({
    name: "langchain-docs-retrieval",
    description:
      "Tool for retrieving from document",
    chain,
    returnDirect: true,
  });

  const searchTool = new SerpAPI(process.env.SERPAPI_API_KEY, {
    location: "Columbus,Ohio,United States",
    hl: "en",
    gl: "us",
  });

  // const agentExecutor = await initializeAgentExecutorWithOptions([qaTool, searchTool], model, {
  //   agentType: "zero-shot-react-description",
  //   verbose: true,
  // });
  const agentExecutor = PlanAndExecuteAgentExecutor.fromLLMAndTools({
    llm: model,
    tools: [qaTool, searchTool],
  });

  const result = await agentExecutor.call({input: 
  `I am a react and javascript developer and am interested in creating ai-powered applications using langchain.
  I want to load all pages in a website and then ask the chain questions about data specific to that website.
  I know that loading all that data and creating the vector store takes a long time, so I want to 
  save the vector store to a file and then load it on subsequent runs of the tool so I don't have to re-load
  all the web pages.  Please write me a javascript function that uses langchain to do this. Your answer should be actual code.
  `
  });
  console.log(`Got output ${result.output}`);
};