import OpenAI from "openai";
import type { AssistantStream } from "openai/lib/AssistantStream";
import type { Channel, Event, MessageResponse, StreamChat } from "stream-chat";

export class OpenAIResposnseHandler {
    private message_text = "";
    private chunk_counter = 0;
    private run_id = "";
    private is_done = false;
    private last_update_time = 0;

    constructor(
        private readonly openai: OpenAI,
        private readonly openAiThread: OpenAI.Beta.Threads.Thread,
        private readonly assistantStream: AssistantStream,
        private readonly chatClient: StreamChat,
        private readonly channel: Channel,
        private readonly message: MessageResponse,
        private readonly onDispose: () => void,
    ) {
        this.chatClient.on("ai_indicator.stop", this.handleStopGenerating);
    }

    run = async () => { }
    dispose = async () => {
        if (this.is_done) {
            return;
        }
        this.is_done = true;
        this.chatClient.off("ai_indicator.stop", this.handleStopGenerating);
        this.onDispose();

    };

    private handleStopGenerating = async (event: Event) => {
        if (this.is_done || event.message_id != this.message.id) {
            return;
        }
        console.log("Stop Generating For Message", this.message.id);
        if (!this.openai || !this.openAiThread || !this.run_id) {
            return;
        }

        try {
            await this.openai.beta.threads.runs.cancel(
                this.run_id,                         // runId
                { thread_id: this.openAiThread.id }  // params
            );

        } catch (error) {
            console.error("Error Canceling Run", error);
        }

        await this.channel.sendEvent({
            type: "ai_indicator.clear",
            cid: this.message.cid,
            message_id: this.message.id
        })
        await this.dispose();
    };

    private handleStreamEvent = async (event: OpenAI.Beta.Assistants.AssistantStreamEvent) => {
        const { cid, id } = this.message;

        if (event.event === "thread.run.created") {
            this.run_id = event.data.id;
        } else if (event.event === "thread.message.delta") {
            const textDelta = event.data.delta.content?.[0]
            if (textDelta?.type === "text" && textDelta.text) {
                this.message_text += textDelta.text.value || "";
                const now = Date.now();
                if (now - this.last_update_time > 1000) {
                    this.chatClient.partialUpdateMessage(id, {
                        set: { text: this.message_text }
                    })
                    this.last_update_time = now;
                }
                this.chunk_counter += 1;
            }
        } else if (event.event === "thread.message.completed") {
            this.chatClient.partialUpdateMessage(id, {
                set: {
                    text: event.data.content[0].type === "text"
                        ? event.data.content[0].text.value
                        : this.message_text
                },
            });
            this.channel.sendEvent({
                type: "ai_indicator.clear",
                cid: cid,
                message_id: id
            })

        } else if (event.event === "thread.run.step.created") {
            if (event.data.step_details.type === "message_creation") {
                this.channel.sendEvent({
                    type: "ai_indicator.update",
                    ai_state: 'AI_STATE_GENERATING',
                    cid: cid,
                    message_id: id
                })
            }
        }

    };

    private handleStreamError = async (error: Error) => {
        if (this.is_done) {
            return;
        }
        await this.channel.sendEvent({
            type: "ai_indicator.update",
            ai_state: 'AI_STATE_ERROR',
            cid: this.message.cid,
            message_id: this.message.id
        })
        await this.chatClient.partialUpdateMessage(this.message.id, {
            set: {
                text: error.message ?? "ErrorGenerating The Message",
                message: error.toString()
            }
        })
        await this.dispose();
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

            if (response.ok) {
                const errorText = await response.text();
                console.log(`Tavily Search Failed For Query: ${query} :`, errorText);

                return JSON.stringify({
                    error: `Search Failed with Status Code: ${response.status}`,
                    details: errorText,
                });
            }

            const data = await response.json();
            console.log(`Tavily Search Successfull For Query: ${query}`)

            return JSON.stringify(data);

        } catch (error) {
            console.error(`An Exception Occurred During Web Search For Query: ${query}`);

            return JSON.stringify({
                error: "An Exception Occurred During Web Search",
            });
        }

    };
}
