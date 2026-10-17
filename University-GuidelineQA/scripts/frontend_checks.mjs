// Exercise the actual frontend against a running model without a browser.
// This is a DOM harness, not a substitute for visual browser QA.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';
import {Blob} from 'node:buffer';

export async function testFrontend(packagePath, baseUrl) {
  const source=await fs.readFile(packagePath+'/web/app.js','utf8');
  const prompt=JSON.parse(await fs.readFile(packagePath+'/web/gaapo/prompt.json','utf8'));
  const elements=new Map();
  function element(id){
    const item={id,value:'',textContent:'',className:'',hidden:false,disabled:false,readOnly:false,
      listeners:{},dataset:{},scrollTop:0,scrollHeight:0,
      addEventListener(name,fn){this.listeners[name]=fn;},focus(){},click(){this.clicked=true;}};
    item.classList={remove(name){item.className=item.className.split(' ').filter(x=>x!==name).join(' ');},
      add(name){item.className+=' '+name;}};
    return item;
  }
  for(const id of ['question-form','question','answer','ask-button','stop-button','copy-button','download-button','timing','error','status','status-text','answer-length','generation-mode','question-label','mode-help','answer-title','selected-prompt','output-requirements','prompt-provenance'])elements.set(id,element(id));
  elements.get('ask-button').disabled=true;
  elements.get('generation-mode').value='gaapo';
  elements.get('answer-length').value='128';
  let copied='', downloaded=null, missingPrompt=false;
  const requests=[];
  const suggestions=[element('plain-suggestion'),element('gaapo-suggestion')];
  suggestions[0].dataset={mode:'plain',question:'What is heart failure?'};
  suggestions[1].dataset={mode:'gaapo',question:'Create one patient-education scene explaining what heart failure is. Highlight Heart failure on screen.'};
  const testURL={createObjectURL(blob){downloaded=blob;return 'blob:test';},revokeObjectURL(){}};
  const sandbox={document:{querySelector(selector){return elements.get(selector.slice(1));},querySelectorAll(){return suggestions;},createElement(){return element('link');}},
    fetch:(path,options)=>{
      if(missingPrompt&&path==='/gaapo/prompt.json')return Promise.resolve(new Response('{}',{status:404}));
      if(path==='/v1/chat/completions')requests.push(JSON.parse(options.body));
      return fetch(baseUrl+path,options);
    },TextDecoder,Uint8Array,AbortController,Blob,
    URL:testURL,performance,navigator:{clipboard:{async writeText(text){copied=text;}}},
    setInterval(){},setTimeout(){},console};
  vm.createContext(sandbox);new vm.Script(source).runInContext(sandbox);
  for(let i=0;i<100&&elements.get('ask-button').disabled;i++)await new Promise(r=>setTimeout(r,20));
  assert.equal(elements.get('status-text').textContent,'Model ready');
  assert.equal(elements.get('generation-mode').value,'gaapo');
  assert.equal(elements.get('selected-prompt').textContent,prompt.prompt_text);
  assert.equal(suggestions[0].hidden,true);
  assert.equal(suggestions[1].hidden,false);
  elements.get('generation-mode').value='plain';elements.get('generation-mode').listeners.change();
  assert.equal(elements.get('ask-button').textContent,'Ask model');
  assert.equal(suggestions[0].hidden,false);
  suggestions[0].listeners.click();
  assert.equal(elements.get('question').value,'What is heart failure?');
  const submission=elements.get('question-form').listeners.submit({preventDefault(){}});
  assert.equal(elements.get('question').readOnly,true);
  assert.equal(elements.get('generation-mode').disabled,true);
  await submission;
  assert(elements.get('answer').textContent.length>50,'Streamed model answer should be visible');
  assert.equal(elements.get('stop-button').hidden,true);
  assert.equal(elements.get('error').hidden,true);
  assert(elements.get('timing').textContent.includes('first text'));
  assert.deepEqual(requests[0].messages,[{role:'user',content:'What is heart failure?'}]);
  await elements.get('copy-button').listeners.click();
  assert.equal(copied,elements.get('answer').textContent);
  elements.get('question').value='Changed question after generation';
  elements.get('download-button').listeners.click();
  const download=await downloaded.text();
  assert(download.includes('Question\nWhat is heart failure?'),'Download must retain original question');
  assert(download.includes(copied));
  elements.get('generation-mode').value='gaapo';elements.get('generation-mode').listeners.change();
  suggestions[1].listeners.click();elements.get('answer-length').value='256';
  await elements.get('question-form').listeners.submit({preventDefault(){}});
  assert.deepEqual(requests[1].messages,[{role:'system',content:prompt.system_prompt},{role:'user',content:suggestions[1].dataset.question}]);
  assert(elements.get('answer').textContent.length>30,'GAAPO mode should show the streamed response');
  assert.equal(elements.get('error').hidden,true);
  const gaapoTiming=elements.get('timing').textContent;
  elements.get('question').value='Edited after storyboard generation';
  elements.get('generation-mode').value='plain';elements.get('generation-mode').listeners.change();
  elements.get('download-button').listeners.click();
  const storyboardDownload=await downloaded.text();
  assert(storyboardDownload.includes('Mode: GAAPO best prompt'));
  assert(storyboardDownload.includes(prompt.provenance.sha256));
  assert(storyboardDownload.includes('Request\n'+suggestions[1].dataset.question));
  elements.get('question').value='Explain heart failure.';elements.get('answer-length').value='512';
  const cancelled=elements.get('question-form').listeners.submit({preventDefault(){}});
  await new Promise(r=>setTimeout(r,800));
  elements.get('stop-button').listeners.click();
  await cancelled;
  assert.equal(elements.get('timing').textContent,'Stopped.');
  assert.equal(elements.get('question').readOnly,false);
  assert.equal(elements.get('ask-button').disabled,false);
  assert.equal(elements.get('generation-mode').disabled,false);
  // A missing prompt must block GAAPO instead of silently using plain Q&A.
  missingPrompt=true;requests.length=0;
  elements.get('generation-mode').value='gaapo';
  const failureSandbox={...sandbox};
  vm.createContext(failureSandbox);new vm.Script(source).runInContext(failureSandbox);
  for(let i=0;i<100&&elements.get('status-text').textContent!=='GAAPO prompt unavailable';i++)await new Promise(r=>setTimeout(r,20));
  assert.equal(elements.get('status-text').textContent,'GAAPO prompt unavailable');
  assert.equal(elements.get('ask-button').disabled,true);
  await elements.get('question-form').listeners.submit({preventDefault(){}});
  assert.equal(requests.length,0);
  elements.get('generation-mode').value='plain';elements.get('generation-mode').listeners.change();
  assert.equal(elements.get('ask-button').disabled,false);
  return {passed:['model and prompt readiness','GAAPO mode selected by default','saved prompt displayed','plain Q&A sends no system prompt','GAAPO sends exact saved system prompt','example request selection','live streamed answers in both modes','loading state locks mode','copy answer','download retains original question and GAAPO provenance','stop cancels generation','controls restored after cancellation','missing prompt blocks GAAPO while plain Q&A remains available'],
    gaapo_response_timing:gaapoTiming,
    visual_browser_qa:'Unavailable: no in-app browser connection was exposed in this session.'};
}

if(import.meta.url.endsWith('/frontend_checks.mjs')&&typeof process!=='undefined'&&process.argv?.[1]?.endsWith('frontend_checks.mjs')){
  const result=await testFrontend(process.argv[2]||'.',process.argv[3]||'http://127.0.0.1:7860');
  console.log(JSON.stringify(result,null,2));
}
