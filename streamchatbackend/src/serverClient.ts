import { StreamChat } from "stream-chat";

export const apiKey = process.env.STREAM_API_KEY as string;
export const apiSecret = process.env.STREAM_API_SECRET as string;

if(!apiKey || !apiSecret) {
    throw new Error("Missing Required Environment Variables For STREAAM_API_KEY and STREAM_API_SECRET");
}

export const serverClient = new StreamChat(apiKey, apiSecret);