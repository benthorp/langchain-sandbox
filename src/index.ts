import * as dotenv from "dotenv";
// import { RunExampleLLMChain } from "./SimpleLLMChain.ts";
import { RetrievalQA } from "./RetrievalQA.ts";
// import { RunGitHubRepoLoader } from "./GitHubLoader.js";
// import { RunSampleChat } from './SampleChat.js'

async function main() {
  dotenv.config();

  const res = await RetrievalQA();

  console.log(res);
}

main();