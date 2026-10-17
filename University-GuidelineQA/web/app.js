"use strict";
const form=document.querySelector('#question-form');
const question=document.querySelector('#question');
const answer=document.querySelector('#answer');
const ask=document.querySelector('#ask-button');
const stop=document.querySelector('#stop-button');
const copy=document.querySelector('#copy-button');
const download=document.querySelector('#download-button');
const timing=document.querySelector('#timing');
const error=document.querySelector('#error');
const status=document.querySelector('#status');
const statusText=document.querySelector('#status-text');
const mode=document.querySelector('#generation-mode');
const length=document.querySelector('#answer-length');
const suggestions=[...document.querySelectorAll('[data-question]')];
let controller=null, busy=false, ready=false, modelReady=false, promptData=null, promptError='', answerText='', answeredQuestion='', answeredMode='gaapo', startedAt=0, firstTextAt=null, finishReason=null;

function updateReady(){
  if(busy)return;
  ready=modelReady&&(mode.value==='plain'||promptData!==null);
  ask.disabled=!ready;
  status.className='status'+(ready?' ready':'');
  statusText.textContent=!modelReady?'Model starting':ready?'Model ready':promptError?'GAAPO prompt unavailable':'Loading GAAPO prompt';
}

async function loadPrompt(){
  try{
    const response=await fetch('/gaapo/prompt.json',{cache:'no-store'});
    if(!response.ok)throw new Error('The saved GAAPO prompt file is missing. Extract the complete handover folder.');
    const data=await response.json();
    if(data.schema_version!==1||!data.prompt_text?.trim()||!data.output_requirements?.trim()||!data.provenance?.candidate_id||
      data.system_prompt!==data.prompt_text+'\nOutput requirements: '+data.output_requirements){
      throw new Error('The saved GAAPO prompt is invalid. Extract the complete handover folder.');
    }
    promptData=data;
    document.querySelector('#selected-prompt').textContent=data.prompt_text;
    document.querySelector('#output-requirements').textContent=data.output_requirements;
    document.querySelector('#prompt-provenance').textContent=`Run: ${data.provenance.source_run} · Selected candidate: ${data.provenance.candidate_id}\nPrompt SHA256: ${data.provenance.sha256}`;
  }catch(exception){
    promptError=exception.message;
    document.querySelector('#prompt-provenance').textContent=promptError;
    if(mode.value==='gaapo'){error.textContent=promptError;error.hidden=false;}
  }
  updateReady();
}

function updateMode(){
  const gaapo=mode.value==='gaapo';
  document.querySelector('#question-label').textContent=gaapo?'Describe your storyboard':'Ask a question';
  document.querySelector('#mode-help').textContent=gaapo?
    'Uses the saved GAAPO prompt to request storyboard JSON. Start with one scene for a short response.':
    'Sends your question directly to the fine-tuned GuidelineKG model.';
  question.placeholder=gaapo?'Create one patient-education scene explaining what heart failure is. Highlight Heart failure on screen.':'What would you like to ask about the guidelines?';
  ask.textContent=gaapo?'Generate storyboard':'Ask model';
  suggestions.forEach(button=>button.hidden=button.dataset.mode!==mode.value);
  if(!answerText)document.querySelector('#answer-title').textContent=gaapo?'Generated storyboard':'Model answer';
  if(promptError){error.textContent=promptError;error.hidden=!gaapo;}
  updateReady();
}

mode.addEventListener('change',updateMode);

async function health(){
  if(busy)return;
  try{
    const response=await fetch('/health');
    const data=await response.json();
    modelReady=response.ok && data.status==='ok';
  }catch{modelReady=false;}
  updateReady();
}

suggestions.forEach(button=>button.addEventListener('click',()=>{
  if(busy)return;
  question.value=button.dataset.question;
  question.focus();
}));

function consume(event){
  if(event.error)throw new Error(event.error.message||'The model could not answer. Try a shorter question.');
  for(const choice of event.choices||[]){
    const text=choice.delta?.content||'';
    if(text){
      if(firstTextAt===null)firstTextAt=(performance.now()-startedAt)/1000;
      answerText+=text;
      answer.textContent=answerText;
      answer.scrollTop=answer.scrollHeight;
    }
    finishReason=choice.finish_reason||finishReason;
  }
}

form.addEventListener('submit',async event=>{
  event.preventDefault();
  if(busy||!ready||!question.value.trim())return;
  busy=true;
  answeredQuestion=question.value.trim();answeredMode=mode.value;
  question.readOnly=true;mode.disabled=true;length.disabled=true;
  document.querySelector('#answer-title').textContent=answeredMode==='gaapo'?'Generated storyboard':'Model answer';
  controller=new AbortController();
  ask.disabled=true; stop.hidden=false; copy.hidden=true; download.hidden=true;
  error.hidden=true; answerText=''; answer.textContent=''; answer.className='answer generating';
  timing.textContent='Preparing your answer…';
  status.className='status busy'; statusText.textContent='Answering';
  startedAt=performance.now(); firstTextAt=null; finishReason=null;
  try{
    const messages=answeredMode==='gaapo'?[{role:'system',content:promptData.system_prompt},{role:'user',content:answeredQuestion}]:[{role:'user',content:answeredQuestion}];
    const response=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'guidelinekg',messages,
        temperature:0,max_tokens:Number(length.value),stream:true,
        stream_options:{include_usage:true},repeat_penalty:1.3,repeat_last_n:2048}),signal:controller.signal});
    if(!response.ok){const data=await response.json();throw new Error(data.error?.message||'The model is busy or your question is too long. Please try again.');}
    const reader=response.body.getReader(), decoder=new TextDecoder();
    let buffer='';
    while(true){
      const {value,done}=await reader.read();
      buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
      let newline;
      while((newline=buffer.indexOf('\n'))>=0){
        const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);
        if(line.startsWith('data: ')){const value=line.slice(6).trim();if(value!=='[DONE]')consume(JSON.parse(value));}
      }
      if(done){if(buffer.startsWith('data: ')&&buffer.slice(6).trim()!=='[DONE]')consume(JSON.parse(buffer.slice(6)));break;}
    }
    if(!answerText)throw new Error('The model returned an empty answer. Please try another question.');
    timing.textContent=`${((performance.now()-startedAt)/1000).toFixed(1)}s total`+(firstTextAt!==null?` · first text ${firstTextAt.toFixed(1)}s`:'');
    if(finishReason==='length')timing.textContent+=' · length limit reached';
  }catch(exception){
    if(exception.name==='AbortError')timing.textContent='Stopped.';
    else{error.textContent=exception.message;error.hidden=false;timing.textContent='Answer interrupted.';}
  }finally{
    busy=false; question.readOnly=false; mode.disabled=false; length.disabled=false; controller=null; stop.hidden=true; answer.classList.remove('generating');
    if(!answerText){answer.textContent='Your answer will appear here.';answer.classList.add('empty');}
    copy.hidden=!answerText;download.hidden=!answerText;await health();
  }
});

stop.addEventListener('click',()=>controller?.abort());
copy.addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(answerText);copy.textContent='Copied';setTimeout(()=>copy.textContent='Copy answer',1800);}
  catch{copy.textContent='Select text to copy';}
});
download.addEventListener('click',()=>{
  const provenance=answeredMode==='gaapo'?`Mode: GAAPO best prompt\nRun: ${promptData.provenance.source_run}\nCandidate: ${promptData.provenance.candidate_id}\nPrompt SHA256: ${promptData.provenance.sha256}\n\n`:'Mode: Plain GuidelineKG Q&A\n\n';
  const blob=new Blob([`${provenance}${answeredMode==='gaapo'?'Request':'Question'}\n${answeredQuestion}\n\nModel answer\n${answerText}\n\nBuilt with Llama · GuidelineKG research demonstration\n`],{type:'text/plain;charset=utf-8'});
  const url=URL.createObjectURL(blob), link=document.createElement('a');link.href=url;link.download=answeredMode==='gaapo'?'gaapo-storyboard.txt':'guideline-answer.txt';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
updateMode();loadPrompt();health();setInterval(health,5000);
