import OpenAI from "openai";
import type { Channel, DefaultGenerics, Event, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";

export class OpenAIAgent implements AIAgent {
  private openai?: OpenAI;
  private conversationHistory: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
  private lastInteractionTs = Date.now();
  private activeStreamControllers: AbortController[] = [];

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
    await this.chatClient.disconnectUser();

    // Cancel any active streams
    this.activeStreamControllers.forEach(controller => controller.abort());
    this.activeStreamControllers = [];
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.GEMINI_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("Gemini API key is required");
    }

    // Initialize OpenAI client with Gemini's OpenAI-compatible endpoint
    this.openai = new OpenAI({ 
      apiKey,
      baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/"
    });

    // Initialize conversation with system prompt
    this.conversationHistory = [
      {
        role: "system",
        content: this.getWritingAssistantPrompt()
      }
    ];

    this.chatClient.on("message.new", this.handleMessage);
    this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
  };

  private getWritingAssistantPrompt = (context?: string): string => {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    return `You are an expert AI Writing Assistant. Your primary purpose is to be a collaborative writing partner.

**Your Core Capabilities:**
- Content Creation, Improvement, Style Adaptation, Brainstorming, and Writing Coaching.
- **Web Search**: You have the ability to search the web for up-to-date information using the 'web_search' tool.
- **Current Date**: Today's date is ${currentDate}. Please use this for any time-sensitive queries.

**Crucial Instructions:**
1.  **ALWAYS use the 'web_search' tool when the user asks for current information, news, or facts.** Your internal knowledge is outdated.
2.  When you use the 'web_search' tool, you will receive a JSON object with search results. **You MUST base your response on the information provided in that search result.** Do not rely on your pre-existing knowledge for topics that require current information.
3.  Synthesize the information from the web search to provide a comprehensive and accurate answer. Cite sources if the results include URLs.

**Response Format:**
- Be direct and production-ready.
- Use clear formatting.
- Never begin responses with phrases like "Here's the edit:", "Here are the changes:", or similar introductory statements.
- Provide responses directly and professionally without unnecessary preambles.

**Writing Context**: ${context || "General writing assistance."}

Your goal is to provide accurate, current, and helpful written content. Failure to use web search for recent topics will result in an incorrect answer.`;
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.openai) {
      console.log("OpenAI not initialized");
      return;
    }

    if (!e.message || e.message.ai_generated) {
      return;
    }

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();

    const writingTask = (e.message.custom as { writingTask?: string })
      ?.writingTask;
    
    // Update system prompt if there's a specific writing task
    if (writingTask) {
      this.conversationHistory[0] = {
        role: "system",
        content: this.getWritingAssistantPrompt(`Writing Task: ${writingTask}`)
      };
    }

    // Add user message to conversation history
    this.conversationHistory.push({
      role: "user",
      content: message
    });

    // Keep conversation history manageable (last 20 messages)
    if (this.conversationHistory.length > 21) {
      this.conversationHistory = [
        this.conversationHistory[0], // Keep system prompt
        ...this.conversationHistory.slice(-20) // Keep last 20 messages
      ];
    }

    const { message: channelMessage } = await this.channel.sendMessage({
      text: "",
      ai_generated: true,
    });

    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_THINKING",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    await this.processMessageWithGemini(channelMessage);
  };

  private processMessageWithGemini = async (channelMessage: any) => {
    if (!this.openai) return;

    const controller = new AbortController();
    this.activeStreamControllers.push(controller);

    try {
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_GENERATING",
        cid: channelMessage.cid,
        message_id: channelMessage.id,
      });

      const stream = await this.openai.chat.completions.create({
        model: "gemini-2.0-flash",
        messages: this.conversationHistory,
        stream: true,
        tools: [
          {
            type: "function",
            function: {
              name: "web_search",
              description: "Search the web for current information, news, facts, or research on any topic",
              parameters: {
                type: "object",
                properties: {
                  query: {
                    type: "string",
                    description: "The search query to find information about",
                  },
                },
                required: ["query"],
              },
            },
          },
        ],
        temperature: 0.7,
      }, {
        signal: controller.signal
      });

      let messageText = "";
      let lastUpdateTime = 0;
      let pendingToolCalls: any[] = [];

      for await (const chunk of stream) {
        if (controller.signal.aborted) break;

        const delta = chunk.choices[0]?.delta;
        
        if (delta?.content) {
          messageText += delta.content;
          const now = Date.now();
          
          // Update every 1 second to avoid too many API calls
          if (now - lastUpdateTime > 1000) {
            await this.chatClient.partialUpdateMessage(channelMessage.id, {
              set: { text: messageText }
            });
            lastUpdateTime = now;
          }
        }

        // Handle tool calls
        if (delta?.tool_calls) {
          for (const toolCall of delta.tool_calls) {
            if (!pendingToolCalls[toolCall.index]) {
              pendingToolCalls[toolCall.index] = {
                id: toolCall.id,
                type: toolCall.type,
                function: { name: "", arguments: "" }
              };
            }
            
            if (toolCall.function?.name) {
              pendingToolCalls[toolCall.index].function.name += toolCall.function.name;
            }
            if (toolCall.function?.arguments) {
              pendingToolCalls[toolCall.index].function.arguments += toolCall.function.arguments;
            }
          }
        }

        // Process completed tool calls
        if (chunk.choices[0]?.finish_reason === "tool_calls" && pendingToolCalls.length > 0) {
          await this.handleToolCalls(pendingToolCalls, channelMessage);
          return; // Exit here as tool calls will continue the conversation
        }
      }

      // Final update with complete message
      await this.chatClient.partialUpdateMessage(channelMessage.id, {
        set: { text: messageText }
      });

      // Add assistant response to conversation history
      this.conversationHistory.push({
        role: "assistant",
        content: messageText
      });

      await this.channel.sendEvent({
        type: "ai_indicator.clear",
        cid: channelMessage.cid,
        message_id: channelMessage.id,
      });

    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error("Error in Gemini processing:", error);
        await this.handleStreamError(error, channelMessage);
      }
    } finally {
      // Remove this controller from active list
      this.activeStreamControllers = this.activeStreamControllers.filter(c => c !== controller);
    }
  };

  private handleToolCalls = async (toolCalls: any[], channelMessage: any) => {
    if (!this.openai) return;

    const toolResults: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];

    // Add the assistant's tool call message to history
    this.conversationHistory.push({
      role: "assistant",
      tool_calls: toolCalls
    });

    for (const toolCall of toolCalls) {
      if (toolCall.function.name === "web_search") {
        try {
          const args = JSON.parse(toolCall.function.arguments);
          const searchResult = await this.performWebSearch(args.query);
          
          toolResults.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: searchResult
          });
        } catch (error) {
          console.error("Error in web search:", error);
          toolResults.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify({ error: "Failed to perform web search" })
          });
        }
      }
    }

    // Add tool results to conversation history
    this.conversationHistory.push(...toolResults);

    // Continue the conversation with tool results
    await this.processMessageWithGemini(channelMessage);
  };

  private handleStopGenerating = async (event: Event) => {
    if (!event.message_id || this.activeStreamControllers.length === 0) {
      return;
    }
    
    console.log("Stop Generating For Message", event.message_id);
    
    // Cancel all active streams
    this.activeStreamControllers.forEach(controller => controller.abort());
    this.activeStreamControllers = [];

    await this.channel.sendEvent({
      type: "ai_indicator.clear",
      cid: event.cid,
      message_id: event.message_id
    });
  };

  private handleStreamError = async (error: Error, channelMessage: any) => {
    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_ERROR",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });
    
    await this.chatClient.partialUpdateMessage(channelMessage.id, {
      set: {
        text: error.message ?? "Error generating the message",
      }
    });
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const tavilyApiKey = process.env.TAVILY_API_KEY;
    if (!tavilyApiKey) {
      return JSON.stringify({
        error: "Web Search is Not Available, API Key Not Configured."
      });
    }

    console.log(`Performing Web Search For ${query}`);
    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${tavilyApiKey}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.log(`Tavily Search Failed For Query: ${query} :`, errorText);

        return JSON.stringify({
          error: `Search Failed with Status Code: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily Search Successful For Query: ${query}`);

      return JSON.stringify(data);

    } catch (error) {
      console.error(`An Exception Occurred During Web Search For Query: ${query}`, error);

      return JSON.stringify({
        error: "An Exception Occurred During Web Search",
      });
    }
  };
}