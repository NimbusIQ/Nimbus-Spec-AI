/** * @license * SPDX-License-Identifier: Apache-2.0 */
import { createApp, ref, defineComponent, onMounted, onUnmounted, computed, watch, nextTick } from 'vue';
import { GoogleGenAI, LiveServerMessage, Modality, Session, Type } from '@google/genai';

const DEFAULT_DIALOG_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025';
const QUIET_THRESHOLD = 0.01; 
const QUIET_DURATION = 2000; 
const EXTENDED_QUIET_DURATION = 10000;
const KEY_URL = 'key.jpeg'; // Placeholder
const PRELOAD_URL = 'preload.png'; // Placeholder

declare global {
  interface Window {
    webkitAudioContext: typeof AudioContext;
    aistudio?: AIStudio;
  }
  interface AIStudio {
    hasSelectedApiKey: () => Promise<boolean>;
    openSelectKey: () => Promise<void>;
  }
}

// --- Data Constants ---

const CHARACTER_ATTRIBUTES: Record<string, any> = {
  'dog': { 
    name: 'Rowan "Barn" Beagle', 
    emoji: '🐶', 
    visualDescriptor: 'A beagle with floppy ears, a wet black nose, and an alert expression. Wears a small detective-style hat.',
    trait: 'You are a loyal, curious detective dog.'
  },
  'cat': { 
    name: 'Shiloh "Silky" Siamese', 
    emoji: '🐱', 
    visualDescriptor: 'A sleek Siamese cat with striking blue eyes. Wears a stylish collar.',
    trait: 'You are a sophisticated, slightly sarcastic but caring cat.'
  },
  'robot': { 
    name: 'R0-B0', 
    emoji: '🤖', 
    visualDescriptor: 'A cute, round robot with glowing eyes and a metallic finish.',
    trait: 'You are a helpful, logical, and enthusiastic robot assistant.'
  },
  'alien': { 
    name: 'Zorp', 
    emoji: '👽', 
    visualDescriptor: 'A friendly green alien with large black eyes and a small spacesuit.',
    trait: 'You are a curious explorer from another planet learning about Earth.'
  }
};

const MOOD_ATTRIBUTES: Record<string, any> = {
  'Happy': { 
    emoji: '😊', 
    visualDescriptor: 'Beaming smile with sparkling eyes, body bouncing with energy.',
    voiceInstruction: 'Speak in a cheerful, energetic, and upbeat tone.'
  },
  'Sad': { 
    emoji: '😭', 
    visualDescriptor: 'Streaming tears, slumped shoulders, head hanging low.',
    voiceInstruction: 'Speak in a slow, melancholic, and soft tone.'
  },
  'Cool': { 
    emoji: '😎', 
    visualDescriptor: 'Wearing sunglasses, relaxed posture, looking effortless.',
    voiceInstruction: 'Speak in a laid-back, smooth, and confident tone.'
  },
  'Surprised': { 
    emoji: '😲', 
    visualDescriptor: 'Eyes wide, mouth open, jumping slightly.',
    voiceInstruction: 'Speak in an excited, breathless, and high-pitched tone.'
  }
};

const VISUAL_ACCESSORIES: Record<string, string[]> = {
  'Default': ['a colorful scarf', 'a shiny badge', 'a small hat', 'a backpack']
};

// --- Helpers ---

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): { data: string; mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- Components ---

const LiveAudioComponent = defineComponent({
  props: {
    initialMessage: { type: String, default: "Hello!" },
    voiceName: { type: String, default: 'Kore' },
    systemInstruction: { type: String, default: 'You are a helpful assistant.' }
  },
  emits: ['speaking-start', 'error'],
  setup(props, { emit, expose }) {
    const isRecording = ref(false);
    const audioContexts = ref<{ input?: AudioContext; output?: AudioContext }>({});
    const volume = ref(0);
    let session: Session | null = null;
    let nextStartTime = 0;
    let stream: MediaStream | null = null;
    
    const cleanup = () => {
      isRecording.value = false;
      session?.close();
      stream?.getTracks().forEach(t => t.stop());
      audioContexts.value.input?.close();
      audioContexts.value.output?.close();
      session = null;
    };

    const startSession = async () => {
      if (isRecording.value) return;
      
      try {
        const inputCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const outputCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
        audioContexts.value = { input: inputCtx, output: outputCtx };
        
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = inputCtx.createMediaStreamSource(stream);
        const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
        
        // Volume visualization
        const analyser = inputCtx.createAnalyser();
        analyser.fftSize = 32;
        source.connect(analyser);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const updateVolume = () => {
          if (!isRecording.value) return;
          analyser.getByteFrequencyData(dataArray);
          const avg = dataArray.reduce((a, b) => a + b) / dataArray.length;
          volume.value = avg / 255;
          requestAnimationFrame(updateVolume);
        };
        updateVolume();

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        // Use promise pattern to avoid race condition in onopen
        const sessionPromise = ai.live.connect({
          model: DEFAULT_DIALOG_MODEL,
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: props.voiceName } } },
            systemInstruction: props.systemInstruction,
          },
          callbacks: {
            onopen: () => {
              isRecording.value = true;
              scriptProcessor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmBlob = createBlob(inputData);
                sessionPromise.then(s => s.sendRealtimeInput({ media: pcmBlob }));
              };
              source.connect(scriptProcessor);
              scriptProcessor.connect(inputCtx.destination);
              sessionPromise.then(s => s.sendClientContent({ turns: [{ parts: [{ text: props.initialMessage }] }], turnComplete: true }));
            },
            onmessage: async (msg: LiveServerMessage) => {
              const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
              if (audioData) {
                emit('speaking-start');
                nextStartTime = Math.max(nextStartTime, outputCtx.currentTime);
                const audioBuffer = await decodeAudioData(decode(audioData), outputCtx, 24000, 1);
                const source = outputCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(outputCtx.destination);
                source.start(nextStartTime);
                nextStartTime += audioBuffer.duration;
              }
            },
            onerror: (e) => {
              console.error(e);
              emit('error', e);
              cleanup();
            },
            onclose: () => cleanup()
          }
        });

        session = await sessionPromise;

      } catch (err) {
        console.error(err);
        cleanup();
      }
    };

    onUnmounted(cleanup);
    expose({ startSession, stopSession: cleanup });

    return { isRecording, startSession, cleanup, volume };
  },
  template: `
    <div class="flex items-center gap-4">
      <button @click="isRecording ? cleanup() : startSession()" 
        :class="['w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg', isRecording ? 'bg-red-500 scale-110' : 'bg-purple text-white hover:bg-purple/90']">
        <span v-if="!isRecording" class="text-2xl">🎤</span>
        <div v-else class="flex gap-1 items-end h-8">
           <div class="w-1 bg-white rounded-full animate-wave" :style="{height: Math.max(20, volume * 100) + '%'}"></div>
           <div class="w-1 bg-white rounded-full animate-wave" style="animation-delay: 0.1s" :style="{height: Math.max(20, volume * 80) + '%'}"></div>
           <div class="w-1 bg-white rounded-full animate-wave" style="animation-delay: 0.2s" :style="{height: Math.max(20, volume * 100) + '%'}"></div>
        </div>
      </button>
      <div class="text-sm font-medium text-slate-600">
        {{ isRecording ? 'Listening...' : 'Tap to Chat' }}
      </div>
    </div>
  `
});

const CharacterImage = defineComponent({
  props: {
    character: { type: String, required: true },
    mood: { type: String, default: '' },
  },
  setup(props) {
    const imageUrl = ref('');
    const isLoading = ref(false);
    const error = ref('');

    const generate = async () => {
      isLoading.value = true;
      error.value = '';
      imageUrl.value = '';
      
      const charInfo = CHARACTER_ATTRIBUTES[props.character] || CHARACTER_ATTRIBUTES['dog'];
      const moodInfo = MOOD_ATTRIBUTES[props.mood] || { visualDescriptor: 'neutral expression' };
      
      const prompt = `A cute claymation style 3d render of ${charInfo.visualDescriptor}. The character is ${moodInfo.visualDescriptor}. Simple shapes, vibrant colors, clean white background, soft lighting, depth of field. High quality, miniature aesthetic.`;

      try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const resp = await ai.models.generateImages({
          model: 'imagen-4.0-generate-001',
          prompt,
          config: { numberOfImages: 1, aspectRatio: '1:1', outputMimeType: 'image/jpeg' }
        });
        if (resp.generatedImages?.[0]?.image?.imageBytes) {
          imageUrl.value = `data:image/jpeg;base64,${resp.generatedImages[0].image.imageBytes}`;
        }
      } catch (e: any) {
        error.value = e.message || 'Failed to generate';
      } finally {
        isLoading.value = false;
      }
    };

    onMounted(generate);
    watch(() => [props.character, props.mood], generate);

    return { imageUrl, isLoading, error, generate };
  },
  template: `
    <div class="relative aspect-square rounded-2xl overflow-hidden bg-slate-100 shadow-inner group">
      <div v-if="isLoading" class="absolute inset-0 flex flex-col items-center justify-center bg-white/50 backdrop-blur-sm z-10">
        <div class="w-10 h-10 border-4 border-purple border-t-transparent rounded-full animate-spin"></div>
      </div>
      <div v-if="error" class="absolute inset-0 flex items-center justify-center p-4 text-center text-red-500 text-sm bg-red-50">
        {{ error }}
        <button @click="generate" class="block mt-2 text-xs underline">Retry</button>
      </div>
      <img v-if="imageUrl" :src="imageUrl" class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" />
      <div v-else class="w-full h-full flex items-center justify-center text-4xl">✨</div>
    </div>
  `
});

// --- Feature Components ---

const StudioTool = defineComponent({
  setup() {
    const prompt = ref('');
    const aspectRatio = ref('1:1');
    const result = ref('');
    const loading = ref(false);

    const generate = async () => {
      if (!prompt.value) return;
      loading.value = true;
      result.value = '';
      try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const resp = await ai.models.generateImages({
          model: 'imagen-4.0-generate-001',
          prompt: prompt.value,
          config: { numberOfImages: 1, aspectRatio: aspectRatio.value, outputMimeType: 'image/jpeg' }
        });
        if (resp.generatedImages?.[0]) {
          result.value = `data:image/jpeg;base64,${resp.generatedImages[0].image.imageBytes}`;
        }
      } catch (e) {
        console.error(e);
        alert('Generation failed');
      } finally {
        loading.value = false;
      }
    };

    return { prompt, aspectRatio, result, loading, generate };
  },
  template: `
    <div class="space-y-4 h-full flex flex-col">
      <div class="flex gap-2">
        <select v-model="aspectRatio" class="p-3 rounded-xl border border-purple/20 bg-white/50 focus:outline-none">
          <option value="1:1">1:1 Square</option>
          <option value="16:9">16:9 Landscape</option>
          <option value="9:16">9:16 Portrait</option>
          <option value="4:3">4:3 Standard</option>
          <option value="3:4">3:4 Portrait</option>
        </select>
        <input v-model="prompt" placeholder="Describe your image..." class="flex-1 p-3 rounded-xl border border-purple/20 bg-white/50 focus:outline-none focus:border-purple" @keyup.enter="generate">
        <button @click="generate" :disabled="loading" class="px-6 py-3 bg-purple text-white rounded-xl font-bold hover:bg-purple/90 disabled:opacity-50">
          {{ loading ? '...' : 'Create' }}
        </button>
      </div>
      <div class="flex-1 rounded-2xl bg-slate-50/50 border border-white flex items-center justify-center overflow-hidden relative">
         <img v-if="result" :src="result" class="max-w-full max-h-full object-contain shadow-lg rounded-lg" />
         <div v-else-if="loading" class="animate-pulse text-purple text-lg font-bold">Generating artwork...</div>
         <div v-else class="text-slate-400">Enter a prompt to start</div>
      </div>
    </div>
  `
});

const EditorTool = defineComponent({
  setup() {
    const prompt = ref('');
    const imageFile = ref<File | null>(null);
    const imagePreview = ref('');
    const result = ref('');
    const loading = ref(false);

    const onFileChange = (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        imageFile.value = file;
        const reader = new FileReader();
        reader.onload = (e) => imagePreview.value = e.target?.result as string;
        reader.readAsDataURL(file);
      }
    };

    const edit = async () => {
      if (!prompt.value || !imagePreview.value) return;
      loading.value = true;
      try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const base64Data = imagePreview.value.split(',')[1];
        
        // Using generateContent for editing with gemini-2.5-flash-image ("Nano Banana")
        const resp = await ai.models.generateContent({
          model: 'gemini-2.5-flash-image',
          contents: {
            parts: [
              { inlineData: { mimeType: 'image/jpeg', data: base64Data } },
              { text: prompt.value }
            ]
          }
        });

        // Find image in response
        let foundImage = false;
        for (const part of resp.candidates?.[0]?.content?.parts || []) {
           if (part.inlineData) {
             result.value = `data:image/png;base64,${part.inlineData.data}`;
             foundImage = true;
             break;
           }
        }
        if (!foundImage && resp.text) {
           alert('Model returned text instead of image: ' + resp.text);
        }

      } catch (e) {
        console.error(e);
        alert('Edit failed');
      } finally {
        loading.value = false;
      }
    };

    return { prompt, onFileChange, imagePreview, result, loading, edit };
  },
  template: `
    <div class="h-full flex flex-col space-y-4">
      <div class="flex gap-4 p-4 bg-white/40 rounded-xl">
        <div class="w-1/3 space-y-2">
           <label class="block text-sm font-bold text-slate-600">Original Image</label>
           <div class="relative aspect-square bg-slate-100 rounded-lg overflow-hidden border border-dashed border-slate-300 flex items-center justify-center cursor-pointer hover:bg-slate-50">
             <input type="file" accept="image/*" @change="onFileChange" class="absolute inset-0 opacity-0 cursor-pointer" />
             <img v-if="imagePreview" :src="imagePreview" class="w-full h-full object-cover" />
             <span v-else class="text-slate-400 text-sm">Upload</span>
           </div>
        </div>
        <div class="flex-1 space-y-2 flex flex-col">
           <label class="block text-sm font-bold text-slate-600">Instructions</label>
           <textarea v-model="prompt" class="flex-1 p-3 rounded-lg border border-purple/20 bg-white/80 resize-none focus:outline-none" placeholder="e.g., Make it look like a sketch, remove the background..."></textarea>
           <button @click="edit" :disabled="loading || !imagePreview" class="w-full py-3 bg-purple text-white rounded-lg font-bold hover:bg-purple/90 disabled:opacity-50">
             {{ loading ? 'Editing...' : 'Apply Magic' }}
           </button>
        </div>
      </div>
      <div class="flex-1 bg-slate-50/50 rounded-xl border border-white flex items-center justify-center p-4 overflow-hidden">
        <img v-if="result" :src="result" class="max-w-full max-h-full object-contain rounded-lg shadow-lg" />
        <div v-else class="text-slate-400">{{ loading ? 'Processing pixel magic...' : 'Result will appear here' }}</div>
      </div>
    </div>
  `
});

const CinemaTool = defineComponent({
  setup() {
    const prompt = ref('');
    const aspectRatio = ref('16:9');
    const loading = ref(false);
    const videoUrl = ref('');
    const status = ref('');

    const generate = async () => {
      // Check for API key for Veo
      if (window.aistudio?.hasSelectedApiKey) {
        const hasKey = await window.aistudio.hasSelectedApiKey();
        if (!hasKey) {
          await window.aistudio.openSelectKey();
          // Assume success after dialog
        }
      }

      loading.value = true;
      videoUrl.value = '';
      status.value = 'Initializing Veo...';
      
      try {
        // Re-init client to ensure key is fresh
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        let op = await ai.models.generateVideos({
          model: 'veo-3.1-fast-generate-preview',
          prompt: prompt.value,
          config: {
            numberOfVideos: 1,
            resolution: '720p',
            aspectRatio: aspectRatio.value
          }
        });
        
        while (!op.done) {
          status.value = 'Rendering frames...';
          await new Promise(r => setTimeout(r, 5000));
          op = await ai.operations.getVideosOperation({ operation: op });
        }

        const uri = op.response?.generatedVideos?.[0]?.video?.uri;
        if (uri) {
          status.value = 'Downloading...';
          const vidRes = await fetch(`${uri}&key=${process.env.API_KEY}`);
          const blob = await vidRes.blob();
          videoUrl.value = URL.createObjectURL(blob);
        } else {
          throw new Error('No video URI returned');
        }

      } catch (e) {
        console.error(e);
        status.value = 'Error: ' + (e.message || 'Generation failed');
      } finally {
        loading.value = false;
      }
    };

    return { prompt, aspectRatio, loading, videoUrl, status, generate };
  },
  template: `
    <div class="h-full flex flex-col gap-4">
      <div class="glass-panel p-4 rounded-xl space-y-4">
        <h3 class="font-bold text-purple">Veo Video Generator</h3>
        <div class="flex gap-2">
           <select v-model="aspectRatio" class="p-2 rounded-lg bg-white/50 border border-purple/10">
             <option value="16:9">Landscape</option>
             <option value="9:16">Portrait</option>
           </select>
           <input v-model="prompt" class="flex-1 p-2 rounded-lg bg-white/50 border border-purple/10" placeholder="Describe a video..." />
           <button @click="generate" :disabled="loading" class="px-4 py-2 bg-purple text-white rounded-lg">Generate</button>
        </div>
      </div>
      <div class="flex-1 bg-black/5 rounded-xl flex items-center justify-center relative overflow-hidden">
        <video v-if="videoUrl" :src="videoUrl" controls loop autoplay class="max-w-full max-h-full rounded-lg"></video>
        <div v-else-if="loading" class="text-center space-y-2">
           <div class="w-8 h-8 border-2 border-purple border-t-transparent rounded-full animate-spin mx-auto"></div>
           <p class="text-slate-500 font-mono text-sm">{{ status }}</p>
        </div>
        <div v-else class="text-slate-400">Video output area</div>
      </div>
      <div class="text-xs text-slate-400 text-center">Powered by Veo 3.1 Fast</div>
    </div>
  `
});

const LensTool = defineComponent({
  setup() {
    const analysis = ref('');
    const loading = ref(false);
    const videoRef = ref<HTMLVideoElement>();
    const stream = ref<MediaStream>();
    const capturedImage = ref('');

    onMounted(async () => {
      try {
        stream.value = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.value) videoRef.value.srcObject = stream.value;
      } catch (e) {
        console.error('Camera access denied');
      }
    });

    onUnmounted(() => {
      stream.value?.getTracks().forEach(t => t.stop());
    });

    const captureAndAnalyze = async () => {
      if (!videoRef.value) return;
      
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.value.videoWidth;
      canvas.height = videoRef.value.videoHeight;
      canvas.getContext('2d')?.drawImage(videoRef.value, 0, 0);
      const base64 = canvas.toDataURL('image/jpeg').split(',')[1];
      capturedImage.value = canvas.toDataURL('image/jpeg');
      
      loading.value = true;
      analysis.value = '';
      
      try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const resp = await ai.models.generateContent({
          model: 'gemini-3-pro-preview',
          contents: {
            parts: [
              { inlineData: { mimeType: 'image/jpeg', data: base64 } },
              { text: 'Analyze this image in detail. Describe what you see, identify objects, and explain the context.' }
            ]
          }
        });
        analysis.value = resp.text || 'No analysis available.';
      } catch (e) {
        analysis.value = 'Error analyzing image.';
      } finally {
        loading.value = false;
      }
    };

    return { videoRef, analysis, loading, captureAndAnalyze, capturedImage };
  },
  template: `
    <div class="h-full flex gap-4">
      <div class="w-1/2 flex flex-col gap-4">
         <div class="relative flex-1 bg-black rounded-xl overflow-hidden">
            <video ref="videoRef" autoplay playsinline muted class="absolute inset-0 w-full h-full object-cover"></video>
            <div class="absolute bottom-4 left-0 right-0 flex justify-center">
               <button @click="captureAndAnalyze" class="w-16 h-16 bg-white rounded-full border-4 border-slate-200 shadow-lg hover:scale-105 transition-transform"></button>
            </div>
         </div>
      </div>
      <div class="w-1/2 flex flex-col gap-4">
         <div class="h-1/3 bg-slate-100 rounded-xl overflow-hidden border border-white">
            <img v-if="capturedImage" :src="capturedImage" class="w-full h-full object-cover" />
            <div v-else class="w-full h-full flex items-center justify-center text-slate-400">No Capture</div>
         </div>
         <div class="flex-1 glass-panel p-4 rounded-xl overflow-y-auto font-mono text-sm leading-relaxed whitespace-pre-wrap">
            <div v-if="loading" class="animate-pulse">Analyzing visual data...</div>
            <div v-else>{{ analysis || 'Ready to scan.' }}</div>
         </div>
      </div>
    </div>
  `
});

const ScoutTool = defineComponent({
  setup() {
    const query = ref('');
    const useMaps = ref(false);
    const results = ref<{text: string, chunks: any[]}>({ text: '', chunks: [] });
    const loading = ref(false);

    const search = async () => {
      if (!query.value) return;
      loading.value = true;
      try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        // Get location for maps
        let location = undefined;
        if (useMaps.value) {
           try {
             const pos = await new Promise<GeolocationPosition>((resolve, reject) => {
               navigator.geolocation.getCurrentPosition(resolve, reject);
             });
             location = {
               latitude: pos.coords.latitude,
               longitude: pos.coords.longitude
             };
           } catch (e) { console.warn('Location denied'); }
        }

        const tools = useMaps.value ? [{ googleMaps: {} }] : [{ googleSearch: {} }];
        const toolConfig = location ? { retrievalConfig: { latLng: location } } : undefined;

        const resp = await ai.models.generateContent({
          model: 'gemini-2.5-flash',
          contents: query.value,
          config: {
            tools,
            toolConfig: toolConfig as any
          }
        });
        
        results.value = {
          text: resp.text || '',
          chunks: resp.candidates?.[0]?.groundingMetadata?.groundingChunks || []
        };
      } catch (e) {
        console.error(e);
        results.value.text = 'Search failed.';
      } finally {
        loading.value = false;
      }
    };

    return { query, useMaps, results, loading, search };
  },
  template: `
    <div class="h-full flex flex-col gap-4">
      <div class="flex gap-2 bg-white/40 p-2 rounded-xl">
         <button @click="useMaps = !useMaps" class="p-2 rounded-lg transition-colors" :class="useMaps ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'">
            {{ useMaps ? '📍 Maps' : '🔍 Search' }}
         </button>
         <input v-model="query" @keyup.enter="search" placeholder="Search for anything..." class="flex-1 bg-transparent focus:outline-none px-2" />
         <button @click="search" class="bg-purple text-white px-4 rounded-lg">Go</button>
      </div>
      
      <div class="flex-1 glass-panel p-6 rounded-xl overflow-y-auto">
         <div v-if="loading" class="space-y-2">
            <div class="h-4 bg-slate-200 rounded w-3/4 animate-pulse"></div>
            <div class="h-4 bg-slate-200 rounded w-1/2 animate-pulse"></div>
         </div>
         <div v-else class="prose max-w-none">
            <div class="whitespace-pre-wrap">{{ results.text }}</div>
            
            <!-- Sources / Chunks -->
            <div v-if="results.chunks.length" class="mt-6 pt-4 border-t border-purple/10">
               <h4 class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Sources</h4>
               <div class="grid gap-2 text-sm">
                  <template v-for="(chunk, i) in results.chunks" :key="i">
                     <a v-if="chunk.web?.uri" :href="chunk.web.uri" target="_blank" class="block p-2 bg-white/50 rounded hover:bg-white transition-colors truncate text-blue-600">
                        {{ chunk.web.title || chunk.web.uri }}
                     </a>
                     <a v-if="chunk.maps?.uri" :href="chunk.maps.uri" target="_blank" class="block p-2 bg-white/50 rounded hover:bg-white transition-colors truncate text-green-600">
                        📍 {{ chunk.maps.title || 'Location Result' }}
                     </a>
                  </template>
               </div>
            </div>
         </div>
      </div>
    </div>
  `
});

// --- Main App ---

const App = defineComponent({
  setup() {
    const activeTab = ref('sadie');
    const sadieChar = ref('dog'); // Default avatar
    const sadieMood = ref('Happy');

    const tabs = [
      { id: 'sadie', icon: '✨', label: 'Sadie' },
      { id: 'studio', icon: '🎨', label: 'Studio' },
      { id: 'editor', icon: '🍌', label: 'Editor' },
      { id: 'cinema', icon: '🎬', label: 'Cinema' },
      { id: 'lens', icon: '👁️', label: 'Lens' },
      { id: 'scout', icon: '🧭', label: 'Scout' }
    ];

    return { activeTab, tabs, sadieChar, sadieMood, CHARACTER_ATTRIBUTES, MOOD_ATTRIBUTES };
  },
  template: `
    <div class="fixed inset-0 flex flex-col md:flex-row p-4 gap-4 text-slate-800 font-sans">
      <!-- Nav -->
      <nav class="glass-panel p-2 rounded-2xl flex md:flex-col items-center justify-between md:justify-start gap-4 md:w-20 z-50">
         <div class="w-12 h-12 bg-purple/10 rounded-xl flex items-center justify-center text-2xl font-bold text-purple mb-0 md:mb-4">S</div>
         <div class="flex md:flex-col gap-2 overflow-x-auto w-full md:w-auto md:overflow-visible no-scrollbar">
            <button v-for="tab in tabs" :key="tab.id" @click="activeTab = tab.id"
              class="p-3 rounded-xl transition-all duration-200 flex items-center justify-center relative group"
              :class="activeTab === tab.id ? 'bg-purple text-white shadow-lg shadow-purple/30' : 'hover:bg-white/60 text-slate-500'">
              <span class="text-2xl">{{ tab.icon }}</span>
              <!-- Tooltip -->
              <span class="absolute left-full ml-4 bg-slate-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap hidden md:block z-50">
                {{ tab.label }}
              </span>
            </button>
         </div>
      </nav>

      <!-- Main Stage -->
      <main class="flex-1 glass-panel rounded-3xl p-6 relative overflow-hidden flex flex-col">
        
        <!-- Header -->
        <header class="mb-6 flex justify-between items-center">
           <h1 class="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple to-blue-500">
             {{ tabs.find(t => t.id === activeTab)?.label }}
           </h1>
           <div class="text-xs font-mono text-slate-400">Sadiestar AI v2.5</div>
        </header>

        <!-- Content Area -->
        <div class="flex-1 relative overflow-y-auto min-h-0">
          
          <!-- SADIE TAB -->
          <div v-if="activeTab === 'sadie'" class="h-full flex flex-col md:flex-row gap-8 items-center justify-center">
             <div class="w-full md:w-1/2 max-w-md aspect-square">
                <CharacterImage :character="sadieChar" :mood="sadieMood" />
             </div>
             <div class="w-full md:w-1/2 max-w-md space-y-6">
                <div class="glass p-6 rounded-2xl space-y-4">
                   <h2 class="text-xl font-bold text-slate-700">Chat with Sadie</h2>
                   <div class="flex gap-2 flex-wrap">
                      <select v-model="sadieChar" class="p-2 rounded-lg bg-white/50 border border-slate-200 text-sm">
                        <option v-for="(v, k) in CHARACTER_ATTRIBUTES" :key="k" :value="k">{{ v.name }}</option>
                      </select>
                      <select v-model="sadieMood" class="p-2 rounded-lg bg-white/50 border border-slate-200 text-sm">
                        <option v-for="(v, k) in MOOD_ATTRIBUTES" :key="k" :value="k">{{ k }}</option>
                      </select>
                   </div>
                   <LiveAudioComponent 
                      initialMessage="Hi! I'm Sadie. Let's create something amazing!" 
                      :voiceName="'Kore'"
                      :systemInstruction="CHARACTER_ATTRIBUTES[sadieChar].trait + ' ' + MOOD_ATTRIBUTES[sadieMood].voiceInstruction"
                   />
                </div>
             </div>
          </div>

          <!-- OTHER TOOLS -->
          <StudioTool v-if="activeTab === 'studio'" />
          <EditorTool v-if="activeTab === 'editor'" />
          <CinemaTool v-if="activeTab === 'cinema'" />
          <LensTool v-if="activeTab === 'lens'" />
          <ScoutTool v-if="activeTab === 'scout'" />

        </div>
      </main>
    </div>
  `
});

createApp(App).mount('#app');