# AI Writing Assistant Chat App

A full-stack AI writing assistant built with React, Vite, TypeScript, Express, Stream Chat, Gemini, Tavily web search, Tailwind CSS, and shadcn-style UI components. The project provides a real-time chat interface where users can create writing sessions, talk with an AI assistant, use prompt shortcuts, stream AI responses, stop generation, manage chat history, and get current web-aware writing help through Tavily-backed search.

> Project Type: Full-stack AI chat and writing assistant application<br>
> Frontend: `streamchatfrontend`<br>
> Backend: `streamchatbackend`

---

## Table of Contents

- [Project Overview](#project-overview)
- [How the Project Works](#how-the-project-works)
- [Core Workflows](#core-workflows)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
- [Available Scripts](#available-scripts)
- [Route and API Overview](#route-and-api-overview)
- [Data and Integrations](#data-and-integrations)
- [Security and Validation](#security-and-validation)
- [Deployment Notes](#deployment-notes)
- [Future Improvements](#future-improvements)
- [Project Status](#project-status)

---

## Project Overview

AI Writing Assistant Chat App is designed as a real-time writing workspace. Users enter a username, start writing sessions, send prompts or draft text, and receive streamed AI responses inside a Stream Chat-powered interface.

The application is split into two projects:

- `streamchatfrontend/`: Vite React app for login, chat UI, writing prompt controls, theme switching, Stream Chat connection, and session navigation.
- `streamchatbackend/`: Express TypeScript API for Stream Chat token generation, AI agent lifecycle management, Gemini response streaming, and Tavily-powered web search.

The app combines four major layers:

- Frontend UI components for chat, login, sidebar, prompts, AI status, and messages.
- Stream Chat channels for real-time conversations and writing session persistence.
- Backend AI agent orchestration for starting, stopping, and tracking assistant connections.
- Gemini and Tavily integrations for AI writing responses and current web-aware answers.

---

## How the Project Works

### Application Shell

The frontend is a Vite React application rendered through `src/main.tsx`. It wraps the app with React Router and theme providers, then renders either the login screen or the authenticated chat experience.

The user state is stored in browser `localStorage` under `chat-ai-app-user`. Once a username is entered, the app creates a deterministic user ID from the username using SHA-256 and connects that user to Stream Chat.

### Login Flow

1. A user opens the frontend app.
2. The login form asks for a username.
3. The username is converted into a deterministic Stream user ID.
4. A DiceBear avatar URL is generated for the user.
5. The user object is saved to `localStorage`.
6. The authenticated chat interface is rendered.

### Stream Chat Connection Flow

1. `ChatProvider` reads `VITE_STREAM_API_KEY` and `VITE_BACKEND_URL`.
2. The frontend requests a Stream Chat token from the backend `/token` endpoint.
3. The backend uses the Stream server client and secret to create a time-limited token.
4. `stream-chat-react` connects the user with the returned token.
5. The app can now create, watch, list, and delete Stream Chat channels.

### New Writing Session Flow

1. The user starts a new writing session from the empty state or sidebar.
2. The frontend creates a new Stream Chat `messaging` channel with a UUID.
3. The channel name is derived from the first user message.
4. The frontend calls `/start-ai-agent` with the channel ID.
5. The backend creates an AI bot user for that channel.
6. The bot is added to the Stream channel.
7. The backend creates and initializes an AI agent.
8. The initial user message is sent after the bot joins.
9. The user is navigated to `/chat/:channelId`.

### AI Response Flow

1. The backend AI agent listens for `message.new` events in the active Stream channel.
2. User messages are added to the agent's conversation history.
3. The agent creates an empty AI-generated Stream message.
4. The channel sends AI state events such as thinking and generating.
5. Gemini is called through an OpenAI-compatible client.
6. Streaming chunks are partially written back into the Stream message.
7. The final response is saved to conversation history.
8. The AI indicator is cleared when generation completes.

### Web Search Flow

The AI agent exposes a `web_search` tool to Gemini. When the assistant needs current information, Gemini can request a search query. The backend sends the query to Tavily, receives search results, adds the tool output to conversation history, and continues the AI response using that fresh context.

### Stop Generation Flow

1. While the assistant is generating, the frontend shows a stop button.
2. Clicking it sends an `ai_indicator.stop` event to the Stream channel.
3. The backend AI agent receives the event.
4. Active stream controllers are aborted.
5. The AI indicator is cleared.

---

## Core Workflows

### Writing Workspace

The main chat screen acts as a writing workspace. Before a channel is selected, the user sees a writing-focused empty state with prompt categories for business, content, communication, and creative writing.

### Prompt Shortcuts

The app provides reusable writing prompts such as:

- Fix grammar and spelling.
- Make this more concise.
- Write this more professionally.
- Make it sound more human.
- Summarize key points.
- Continue writing from here.
- Suggest a title.

These prompts can be inserted into the chat input to speed up common writing tasks.

### Session Sidebar

The sidebar lists Stream Chat channels for the current user. Users can open previous writing sessions, start a new one, delete a session, switch between dark and light themes, or log out.

### AI Agent Status

The frontend polls the backend for the current agent state. A channel can be:

- `connected`
- `connecting`
- `disconnected`

Users can manually connect, disconnect, or refresh the AI assistant status for the active writing session.

### Agent Cleanup

The backend keeps AI agents in memory using a `Map`. Agents that remain inactive beyond the configured threshold are disposed automatically, disconnected from Stream Chat, and deleted as bot users.

---

## Features

### Frontend Features

- Username-based login.
- Persistent local user session with `localStorage`.
- Stream Chat client setup with backend token provider.
- Real-time writing sessions.
- Sidebar channel list.
- New writing session creation.
- Channel deletion confirmation dialog.
- Responsive sidebar behavior.
- Dark and light theme switching.
- Toast notifications.
- Custom chat input with auto-resizing textarea.
- Stop generation control while AI is responding.
- Writing prompt categories and toolbar shortcuts.
- AI connection status badge and controls.

### Backend Features

- Express API server.
- Stream Chat server client.
- Secure token generation endpoint.
- AI agent start and stop endpoints.
- Agent status endpoint.
- Per-channel AI bot user creation.
- AI agent cache and pending-agent guard.
- Inactivity-based agent cleanup.
- Gemini streaming responses through OpenAI-compatible API.
- Conversation history management.
- Tavily-backed web search tool.
- Stream Chat AI state events for thinking, generating, error, and clear states.

### AI Writing Features

- Content creation.
- Text improvement.
- Tone adaptation.
- Brainstorming.
- Writing coaching.
- Current-information support through web search.
- Streaming responses inside the chat message.
- Tool-aware responses when Gemini requests web search.

---

## Tech Stack

### Frontend

- React 18
- Vite 5
- TypeScript
- React Router DOM
- Stream Chat React
- Tailwind CSS
- Radix UI primitives
- shadcn-style component structure
- Lucide React
- Framer Motion
- React Hook Form
- Sonner
- js-sha256
- DiceBear avatar URLs

### Backend

- Node.js
- Express
- TypeScript
- Stream Chat server SDK
- OpenAI SDK
- Gemini OpenAI-compatible endpoint
- Tavily Search API
- CORS
- dotenv
- Nodemon

### Tooling

- npm
- ESLint
- Prettier
- TypeScript compiler
- Vite
- Git

---

## Folder Structure

```text
AI ChatBot/
|-- README.md
|-- .gitignore
|-- streamchatbackend/
|   |-- package.json
|   |-- package-lock.json
|   |-- tsconfig.json
|   |-- .env
|   |-- dist/
|   `-- src/
|       |-- index.ts
|       |-- serverClient.ts
|       `-- agents/
|           |-- types.ts
|           |-- createAgent.ts
|           `-- openai/
|               |-- OpenAIAgent.ts
|               `-- OpenAIResponseHandler.ts
`-- streamchatfrontend/
    |-- package.json
    |-- package-lock.json
    |-- vite.config.ts
    |-- tsconfig.json
    |-- tailwind.config.ts
    |-- components.json
    |-- index.html
    |-- .env
    `-- src/
        |-- main.tsx
        |-- App.tsx
        |-- index.css
        |-- providers/
        |   |-- chat-provider.tsx
        |   `-- theme-provider.tsx
        |-- contexts/
        |   `-- theme-context.ts
        |-- hooks/
        |   |-- use-ai-agent-status.tsx
        |   |-- use-mobile.tsx
        |   |-- use-theme.ts
        |   `-- use-toast.ts
        |-- lib/
        |   `-- utils.ts
        `-- components/
            |-- login.tsx
            |-- authenticated-app.tsx
            |-- chat-interface.tsx
            |-- chat-sidebar.tsx
            |-- chat-input.tsx
            |-- chat-message.tsx
            |-- ai-agent-control.tsx
            |-- writing-prompts-toolbar.tsx
            |-- loading-screen.tsx
            `-- ui/
```

---

## Environment Variables

Create separate `.env` files for the backend and frontend. Do not commit real secrets.

### Backend `.env`

```env
PORT=3000

STREAM_API_KEY=<stream-api-key>
STREAM_API_SECRET=<stream-api-secret>

GEMINI_API_KEY=<gemini-api-key>
TAVILY_API_KEY=<tavily-api-key>
```

Required backend variables:

- `STREAM_API_KEY`
- `STREAM_API_SECRET`
- `GEMINI_API_KEY`

Optional backend variable:

- `TAVILY_API_KEY`

`TAVILY_API_KEY` is required only if web search should work. Without it, the AI agent can still respond, but web search tool calls return a configuration error.

### Frontend `.env`

```env
VITE_STREAM_API_KEY=<stream-api-key>
VITE_BACKEND_URL=http://localhost:3000
```

Required frontend variables:

- `VITE_STREAM_API_KEY`
- `VITE_BACKEND_URL`

Important implementation notes:

- The frontend Stream key should match the backend Stream app key.
- The backend URL must point to the running Express API.
- The Stream secret must only exist on the backend.
- Real API keys should never be committed to Git.

---

## Local Setup

### Prerequisites

- Node.js 20 or newer
- npm
- Stream Chat account and app keys
- Gemini API key
- Tavily API key for web search support

### 1. Clone or Open the Project

```bash
cd "AI ChatBot"
```

### 2. Install Backend Dependencies

```bash
cd streamchatbackend
npm install
```

### 3. Install Frontend Dependencies

```bash
cd ../streamchatfrontend
npm install
```

### 4. Configure Backend Environment

Create:

```text
streamchatbackend/.env
```

Use the placeholder values from the [Backend `.env`](#backend-env) section.

### 5. Configure Frontend Environment

Create:

```text
streamchatfrontend/.env
```

Use the placeholder values from the [Frontend `.env`](#frontend-env) section.

### 6. Start the Backend

```bash
cd streamchatbackend
npm run dev
```

The backend runs on:

```text
http://localhost:3000
```

### 7. Start the Frontend

In a second terminal:

```bash
cd streamchatfrontend
npm run dev
```

The frontend usually runs on:

```text
http://localhost:5173
```

### 8. Open the App

Open the frontend URL in the browser, enter a username, create a new writing session, and send a prompt.

---

## Available Scripts

### Backend

```bash
npm run dev
```

Starts the backend in watch mode using Nodemon and TypeScript compilation.

```bash
npm run start
```

Compiles TypeScript and starts the compiled server from `dist/index.js`.

```bash
npm test
```

Placeholder test command. It currently exits with an error because no tests are configured.

### Frontend

```bash
npm run dev
```

Starts the Vite development server.

```bash
npm run build
```

Builds the production frontend bundle.

```bash
npm run build:dev
```

Builds the frontend using development mode.

```bash
npm run lint
```

Runs ESLint checks.

```bash
npm run preview
```

Serves the built frontend locally for preview.

---

## Route and API Overview

### Frontend Routes

| Route | Description |
| --- | --- |
| `/` | Login screen when unauthenticated, or new writing session screen after login. |
| `/chat/:channelId` | Active writing session backed by a Stream Chat channel. |

### Backend API Routes

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health/status response for the AI Writing Assistant server. |
| `POST` | `/token` | Generates a Stream Chat user token for the frontend. |
| `POST` | `/start-ai-agent` | Creates and connects an AI assistant bot for a channel. |
| `POST` | `/stop-ai-agent` | Disconnects and disposes the AI assistant for a channel. |
| `GET` | `/agent-status?channel_id=...` | Returns `connected`, `connecting`, or `disconnected`. |

---

## Data and Integrations

### Stream Chat

Stream Chat powers the real-time chat layer. The frontend uses `stream-chat-react` for chat UI state, channel lists, message lists, active channels, and AI indicators. The backend uses the Stream server SDK to create tokens, upsert AI bot users, add bots to channels, and delete inactive bot users.

### Gemini

The backend uses the OpenAI SDK with Gemini's OpenAI-compatible endpoint:

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

The active model is:

```text
gemini-2.0-flash
```

Gemini generates streamed writing-assistant responses and can request the `web_search` tool for current information.

### Tavily

Tavily is used for web search. When Gemini requests the `web_search` tool, the backend sends the query to Tavily and returns the search results to the model as tool output.

### Browser Local Storage

The frontend stores the logged-in user locally using:

```text
chat-ai-app-user
```

This keeps the user logged in between page refreshes without a separate auth database.

---

## Security and Validation

- Stream Chat user tokens are generated on the backend, not in the browser.
- `STREAM_API_SECRET` is used only in the backend project.
- Frontend environment variables are limited to public Vite values.
- Backend validates required fields such as `userId` and `channel_id`.
- Generated Stream tokens include issued-at and expiration timestamps.
- User IDs are deterministic hashes of usernames instead of raw names.
- AI bot users are created per channel and removed when their agent is disposed.
- Secrets for Stream, Gemini, and Tavily must stay in `.env` files.
- The backend currently allows all CORS origins with `cors({ origin: "*" })`; restrict this in production.

---

## Deployment Notes

### Frontend Deployment

The frontend can be deployed as a Vite static app on platforms such as Vercel, Netlify, or Cloudflare Pages.

Build command:

```bash
npm run build
```

Output directory:

```text
dist
```

Set this production environment variable:

```env
VITE_BACKEND_URL=<deployed-backend-url>
```

### Backend Deployment

The backend can be deployed to a Node.js hosting platform that supports long-running Express servers.

Build/start command:

```bash
npm run start
```

Production notes:

- Set `STREAM_API_KEY`, `STREAM_API_SECRET`, `GEMINI_API_KEY`, and `TAVILY_API_KEY`.
- Set `PORT` according to the hosting provider if required.
- Configure the frontend `VITE_BACKEND_URL` to point to the deployed backend.
- Restrict CORS to the deployed frontend origin.
- Confirm Stream Chat app settings allow the deployed frontend origin.

---

## Future Improvements

- Add proper email/password or OAuth authentication.
- Store users and preferences in a database.
- Add persistent assistant settings per user.
- Add model selection and temperature controls.
- Add stronger rate limiting for token and AI-agent endpoints.
- Restrict backend CORS in production.
- Add request validation with Zod.
- Add tests for backend endpoints and frontend chat flows.
- Fix typos in backend error messages and unused legacy response handler code.
- Add deployment-specific documentation for the chosen hosting platform.
- Add observability for agent lifecycle, token failures, and streaming errors.

---

## Project Status

The project currently includes a working split frontend/backend structure, Stream Chat integration, username-based login, chat session creation, sidebar session management, dark/light theme support, backend Stream token generation, AI agent lifecycle endpoints, Gemini-powered streaming responses, Tavily web search support, prompt shortcut UI, stop-generation handling, and inactive agent cleanup.
