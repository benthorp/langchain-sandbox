import { OpenAI } from "langchain";
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";
import { ChainValues } from "langchain/schema";

export async function RunExampleLLMChain (): Promise<ChainValues>
{
    // LLM
    const model = new OpenAI({
      modelName: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY,
      temperature: 0.9,
    });
  
    // Prompt template
    const template = "What is a good name for a company that makes {product}?";
    const prompt = new PromptTemplate({
      template,
      inputVariables: ["product"],
    });
  
    // Chain
    const chain = new LLMChain({ llm: model, prompt });
  
    // Run
    const res = await chain.call({ product: "colorful socks" });  
    return res;
}