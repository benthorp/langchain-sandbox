import { OpenAI } from 'langchain/llms/openai';
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { VectorDBQAChain, LLMChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { HumanMessagePromptTemplate } from "langchain/prompts";

export async function RunSampleChat() {
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0.9,
  });

  const loader = new CheerioWebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence");

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });

  const docs = await splitter.splitDocuments(await loader.load());

  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  const chain = VectorDBQAChain.fromLLM(model, vectorStore.asRetriever());

  const memory = new BufferMemory();

  const chatChain = new LLMChain({
    llm: chain,
    memory,
  });

  const messagePrompt = new HumanMessagePromptTemplate("What would you like to know about AI?");

  while (true) {
    const input = await messagePrompt.generate();
    const response = await chatChain.call({ input });
    console.log(response.response);
  }
}