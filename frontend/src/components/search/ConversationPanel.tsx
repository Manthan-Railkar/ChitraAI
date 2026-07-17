import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Terminal, Trash2, ChevronRight } from 'lucide-react';
import PixelBlast from '@/components/reactbits/PixelBlast';

interface Message {
  sender: 'user' | 'ai';
  text: string;
  isStreaming?: boolean;
}

interface ConversationPanelProps {
  messages: Message[];
  onSendMessage: (text: string) => void;
  isProcessing: boolean;
  onClearChat: () => void;
  onToggleCollapse: () => void;
  guestSearchCount?: number;
  isGuestSearchLimitReached?: boolean;
}

export const ConversationPanel: React.FC<ConversationPanelProps> = ({
  messages,
  onSendMessage,
  isProcessing,
  onClearChat,
  onToggleCollapse,
  guestSearchCount,
  isGuestSearchLimitReached = false,
}) => {
  const [input, setInput] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

  const suggestedPrompts = [
    'Recommend a mind-bending sci-fi movie',
    "I'm feeling nostalgic tonight",
    'Movies like Interstellar',
    'Hidden psychological thrillers',
  ];

  // Auto-scroll to the bottom of the chat on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;
    onSendMessage(input);
    setInput('');
  };

  const handlePromptClick = (prompt: string) => {
    if (isProcessing) return;
    onSendMessage(prompt);
  };

  return (
    <div className="w-full h-full flex flex-col bg-black/60 backdrop-blur-2xl border-l border-white/[0.06] relative z-20 overflow-hidden">
      {/* Interactive PixelBlast WebGL Background for Chat Panel */}
      <div className="absolute inset-0 z-0 opacity-15 pointer-events-none">
        <PixelBlast
          variant="square"
          pixelSize={4}
          color="#B497CF"
          patternScale={2}
          patternDensity={0.35}
          pixelSizeJitter={0}
          enableRipples
          rippleSpeed={0.4}
          rippleThickness={0.12}
          rippleIntensityScale={1.5}
          liquid={false}
          liquidStrength={0.12}
          liquidRadius={1.2}
          liquidWobbleSpeed={5}
          speed={0.5}
          edgeFade={0.25}
          transparent
        />
      </div>

      {/* Panel Header */}
      <div className="px-5 py-4 border-b border-white/[0.06] flex items-center justify-between bg-white/[0.01] relative z-10">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-rose-500" />
          <span
            className="text-xs font-bold uppercase tracking-widest text-white/80"
            style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
          >
            AI Copilot Workspace
          </span>
        </div>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={onClearChat}
              disabled={isProcessing}
              className="p-1.5 rounded-md border border-white/5 hover:border-white/10 text-white/40 hover:text-white/80 transition-colors disabled:opacity-40"
              title="Clear Chat History"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-md border border-white/5 hover:border-white/10 text-white/40 hover:text-white/80 transition-colors cursor-pointer"
            title="Collapse Workspace"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-5 py-6 flex flex-col gap-4 custom-scrollbar relative z-10">
        {messages.length === 0 ? (
          /* Welcome & Suggested Prompts */
          <div className="flex-1 flex flex-col justify-center gap-6 py-8">
            <div className="flex flex-col gap-2">
              <div className="w-8 h-8 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
                <Sparkles className="w-4 h-4" />
              </div>
              <h3
                className="text-sm font-bold text-white uppercase tracking-wider"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                Start a Discovery Session
              </h3>
              <p
                className="text-xs text-white/40 leading-relaxed"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                Ask ChitraAI for recommendations based on specific plots, emotional tones,
                atmospheres, or compound themes.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <span
                className="text-[10px] font-bold text-white/30 uppercase tracking-widest mb-1"
                style={{ fontFamily: 'Plus Jakarta Sans, sans-serif' }}
              >
                Suggested Prompts
              </span>
              <div className="flex flex-col gap-2.5">
                {suggestedPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => handlePromptClick(prompt)}
                    disabled={isProcessing}
                    className="w-full text-left p-3.5 rounded-xl border border-white/[0.04] bg-white/[0.01] hover:bg-white/[0.03] hover:border-white/10 hover:shadow-[0_4px_20px_rgba(244,63,94,0.02)] text-xs text-white/70 hover:text-white transition-all cursor-pointer select-none leading-normal font-medium"
                    style={{ fontFamily: 'Inter, sans-serif' }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          /* Chat History */
          <div className="flex flex-col gap-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] p-3.5 rounded-2xl text-xs leading-relaxed ${
                    msg.sender === 'user'
                      ? 'bg-rose-500/10 border border-rose-500/20 text-white rounded-tr-none'
                      : 'bg-white/[0.03] border border-white/[0.06] text-white/90 rounded-tl-none'
                  }`}
                  style={{ fontFamily: 'Inter, sans-serif' }}
                >
                  {msg.text}
                  {msg.isStreaming && (
                    <span className="inline-block w-1.5 h-3 bg-rose-500 animate-pulse ml-1 align-middle" />
                  )}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      {/* Message Input Composer */}
      <form
        onSubmit={handleSubmit}
        className="p-4 border-t border-white/[0.06] bg-white/[0.01] relative z-10"
      >
        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-rose-500 via-purple-500 to-blue-500 rounded-xl blur opacity-25 group-focus-within:opacity-50 transition duration-500" />
          <div className="relative flex items-center bg-black/80 border border-white/[0.08] rounded-xl p-1.5 backdrop-blur-xl">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isProcessing}
              placeholder={
                isGuestSearchLimitReached
                  ? 'Free limit reached. Sign in to continue.'
                  : isProcessing
                    ? 'AI is processing...'
                    : 'Describe your perfect film vibe...'
              }
              className="flex-grow bg-transparent border-none text-xs text-white placeholder-white/30 outline-none px-3.5 py-2.5 disabled:opacity-50"
              style={{ fontFamily: 'Inter, sans-serif' }}
            />
            <button
              type="submit"
              disabled={!input.trim() || isProcessing || isGuestSearchLimitReached}
              className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-white text-black hover:bg-rose-500 hover:text-white transition-all disabled:opacity-30 disabled:hover:bg-white disabled:hover:text-black cursor-pointer"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </div>
          {guestSearchCount !== undefined && (
            <p className="mt-2 text-center text-[10px] font-semibold tracking-wide text-white/35">
              Guest Searches: {guestSearchCount} / 5
            </p>
          )}
        </div>
      </form>
    </div>
  );
};

export default ConversationPanel;
