#!/usr/bin/env python3
"""Discord bots that use OpenAI's API to generate responses."""

import asyncio
import json
import os
import random
import sys
import time
import logging
from datetime import datetime
import discord
from discord.ext import commands
from openai import AsyncOpenAI

class Message:
    def __init__(self, author, content, message_id, replying_to=None):
        print("Creating")
        self.author = author
        self.author_id = author.id
        self.message = content
        self.message_id = message_id
        self.replying_to = replying_to.author if replying_to else None
        self.replying_to_replying_to = replying_to.reference.resolved.content if replying_to is not None and replying_to.reference is not None and replying_to.reference.resolved is not None else None
        self.weight = self.calculate_base_weight()
        print("Created.")

    def calculate_base_weight(self):
        base_weight = 0.5 if self.author.bot else 1.0
        if self.replying_to and not self.replying_to.bot:
            base_weight *= 1.2  # Increase weight for replies to actual users
        return base_weight

    def to_dict(self):
        return {
            "author": str(self.author),
            "author_id": str(self.author_id),
            "message": self.message,
            "message_id": self.message_id,
            "replying_to": str(self.replying_to.id) if self.replying_to else None,
            "replying_to_replying_to": self.replying_to_replying_to,
            "weight": self.weight
        }

class GPTBot:
    all_bots = []

    def __init__(self, config):
        # Setup logging
        self.setup_logging(config['name'], config.get('log_color', 'white'))
        self.logger.info(f"Initializing {config['name']} bot")
        
        self.token = config['token']
        self.channel_id = config['channel_id']
        self.model = config['model']
        self.model_name = config['name']
        
        # Load system prompt from file
        with open(config['prompt_file'], 'r') as f:
            self.system_prompt = f.read()
        
        # Initialize variables
        self.message_array = []
        self.MESSAGE_MEMORY = 10
        self.DEBUG = False
        self.DELAY_MIN = 15
        self.DELAY_MAX = 25
        self.time_remaining = 0
        self.NIGHT_MODE_ENABLED = True
        self.NIGHT_DELAY_MIN = 40
        self.NIGHT_DELAY_MAX = 60
        self.bot_mention_pattern = None  # Will be set in on_ready
        
        # Setup bot and client
        self.aclient = AsyncOpenAI(api_key=config['openai_key'])
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix='#', intents=intents)
        
        # Setup event handlers and commands
        self.setup_handlers()
        self.logger.info(f"Bot {config['name']} initialized")
        
        # Add this bot to the list of all bots
        GPTBot.all_bots.append(self)
        
    def setup_logging(self, bot_name, log_color):
        """Setup logging for the bot."""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(bot_name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(f'%(asctime)s - {bot_name} - %(levelname)s - %(message)s')
        
        # File handler
        log_file = f'logs/{bot_name}_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(f'\033[{self.get_color_code(log_color)}m%(asctime)s - {bot_name} - %(levelname)s - %(message)s\033[0m'))
        self.logger.addHandler(console_handler)

    def get_color_code(self, color_name):
        colors = {
            'black': '30',
            'red': '31',
            'green': '32',
            'yellow': '33',
            'blue': '34',
            'magenta': '35',
            'cyan': '36',
            'white': '37'
        }
        return colors.get(color_name.lower(), '37')

    def setup_handlers(self):
        @self.bot.event
        async def on_ready():
            try:
                self.logger.info(f'Bot logged in as {self.bot.user}')
                channel = self.bot.get_channel(self.channel_id)
                await asyncio.wait_for(self.update_message_history(channel), timeout=20)
                await self.update_status()
                self.logger.info(f"Initial messages loaded: {self.message_array}")
                self.bot.loop.create_task(self.send_responses(channel))
                self.bot_mention_pattern = f'<@{self.bot.user.id}>'
            except Exception as e:
                self.logger.error(f"Error in on_ready: {e}")

        @self.bot.event
        async def on_message(message):
            try:
                if message.channel.id == self.channel_id and message.author != self.bot.user and not message.content.startswith('#'):
                    msg = Message(message.author, message.content, message.id,
                                  message.reference.resolved if message.reference else None)
                    
                    # Adjust weight for bot mentions
                    if self.bot_mention_pattern and self.bot_mention_pattern in message.content:
                        msg.weight *= 1.4
                    
                    if len(self.message_array) >= self.MESSAGE_MEMORY:
                        self.message_array.pop(0)
                    self.message_array.append(msg)
                await self.bot.process_commands(message)
            except Exception as e:
                self.logger.error(f"Error in on_message: {e}")

        @self.bot.command(name='reset_history')
        async def reset_history(ctx):
            try:
                self.message_array = []
                await ctx.send("#Message history has been reset.")
                self.logger.info("Message history reset")
            except Exception as e:
                self.logger.error(f"Error in reset_history: {e}")

        @self.bot.command(name='set_memory_size')
        async def set_memory_size(ctx, size: int):
            try:
                if size < 1 or size > 100:
                    await ctx.send("#Invalid memory size. Please choose a number between 1 and 100.")
                    self.logger.warning(f"Invalid memory size attempted: {size}")
                    return
                self.MESSAGE_MEMORY = size
                await self.update_message_history(ctx.channel)
                await ctx.send(f"#Message memory size set to {size}")
                self.logger.info(f"Memory size changed to {size}")
            except Exception as e:
                self.logger.error(f"Error in set_memory_size: {e}")

        @self.bot.command(name='debug')
        async def set_debug(ctx, value: str):
            try:
                if value.lower() not in ['true', 'false']:
                    await ctx.send("#Invalid value. Please use 'true' or 'false'.")
                    return
                self.DEBUG = value.lower() == 'true'
                await ctx.send(f"#Debug mode set to {self.DEBUG}")
                print(f"Debug mode set to {self.DEBUG}")
            except Exception as e:
                self.logger.error(f"Error in set_debug: {e}")

        @self.bot.command(name='set_delay')
        async def set_delay(ctx, min_delay: int, max_delay: int):
            try:
                if min_delay < 1 or max_delay > 60 or min_delay >= max_delay:
                    await ctx.send("#Invalid delay range. Min must be ≥1, max must be ≤60, and min must be less than max.")
                    return
                self.DELAY_MIN = min_delay
                self.DELAY_MAX = max_delay
                await self.update_status()
                await ctx.send(f"#Delay range set to {self.DELAY_MIN}-{self.DELAY_MAX} seconds")
                print(f"Delay range changed to {self.DELAY_MIN}-{self.DELAY_MAX}")
            except Exception as e:
                self.logger.error(f"Error in set_delay: {e}")

        @self.bot.command(name='night_mode')
        async def set_night_mode(ctx, value: str):
            try:
                if value.lower() not in ['true', 'false']:
                    await ctx.send("#Invalid value. Please use 'true' or 'false'.")
                    return
                self.NIGHT_MODE_ENABLED = value.lower() == 'true'
                await ctx.send(f"#Night mode set to {self.NIGHT_MODE_ENABLED}")
                print(f"Night mode set to {self.NIGHT_MODE_ENABLED}")
            except Exception as e:
                self.logger.error(f"Error in set_night_mode: {e}")

        @self.bot.command(name='refresh_prompt')
        async def refresh_prompt(ctx):
            try:
                with open(self.config['prompt_file'], 'r') as f:
                    self.system_prompt = f.read()
                await ctx.send("#System prompt has been refreshed.")
                self.logger.info("System prompt refreshed")
            except Exception as e:
                self.logger.error(f"Error in refresh_prompt: {e}")

        
        # Overwrite the default help command
        self.bot.remove_command('help')

        @self.bot.command(name='help')
        async def help_command(ctx):
            help_text = (
                "#Available commands:\n"
                "#reset_history - Reset the message history.\n"
                "#set_memory_size <size> - Set the message memory size (1-100).\n"
                "#debug <true/false> - Enable or disable debug mode.\n"
                "#set_delay <min_delay> <max_delay> - Set the delay range in seconds.\n"
                "#night_mode <true/false> - Enable or disable night mode.\n"
                "#refresh_prompt - Refresh the system prompt."
            )
            await ctx.send(help_text)

    def prepare_messages_for_ai(self):
        messages = []
        for idx, msg in enumerate(self.message_array):
            weight_multiplier = 1.0
            if msg.author == self.bot.user:
                weight_multiplier = 0.0
            elif idx == len(self.message_array) - 1:  # Newest message
                weight_multiplier = 0.8
            elif idx == len(self.message_array) - 2:
                weight_multiplier = 0.7
            elif idx == len(self.message_array) - 3:
                weight_multiplier = 0.6
            elif idx == len(self.message_array) - 4:
                weight_multiplier = 0.5
            else:
                weight_multiplier = 0.4
            
            msg_dict = msg.to_dict()
            msg_dict['weight'] *= weight_multiplier
            messages.append(msg_dict)
        
        return json.dumps(messages, ensure_ascii=False)

    async def get_ai_response(self, messages):
        try:
            context = self.prepare_messages_for_ai()
            self.logger.info(f"Generating response for context: {context}")
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                presence_penalty=1.5,
                temperature=0.8,
                max_tokens=256,
                response_format={ "type": "json_object" }
            )
            response_json = response.choices[0].message.content
            response_data = json.loads(response_json)
            generated_response = response_data.get("response", "")
            picked_message_id = response_data.get("picked_message", None)
            if picked_message_id:
                picked_message = next((msg for msg in self.message_array if str(msg.message_id) == str(picked_message_id)), None)
                if picked_message:
                    picked_message.weight *= 0.5  # Set weight to half after being responded to
                    self.logger.info(f"Generated response: {generated_response}")
                    self.logger.info(f"Picked message: {picked_message.message}")
                    if generated_response.strip() == picked_message.message.strip() or any(generated_response.strip() == msg.message.strip() for msg in self.message_array):
                        self.logger.warning("Generated response is the same as the picked message or already exists in message history. Regenerating response.")
                        return await self.get_ai_response(messages)  # Regenerate response
                else:
                    self.logger.warning(f"Picked message ID {picked_message_id} not found in message array.")
            else:
                self.logger.warning("No message picked.")
            return {"response": generated_response, "picked_message": picked_message_id}
        except Exception as e:
            err = f"OpenAI API Error: {e}"
            self.logger.error(err)
            return {"response": "", "picked_message": err}

    async def update_status(self):
        activity = discord.Game(f"{self.model_name} | Next: {self.time_remaining}s")
        await self.bot.change_presence(activity=activity)

    async def countdown_status(self, total_seconds):
        self.time_remaining = total_seconds
        while self.time_remaining > 0:
            await self.update_status()
            await asyncio.sleep(3)
            self.time_remaining = max(0, self.time_remaining - 3)
        await self.update_status()

    async def update_message_history(self, channel):
        try:
            messages = []
            self.logger.info(f"Fetching message history for channel: {channel.id}")
            async for message in channel.history(limit=self.MESSAGE_MEMORY):
                self.logger.info(f"Checking message {message.id}")
                if message.content[0] != '#' and message.author != self.bot.user:
                    self.logger.info(f"Adding message: {message.content}")
                    author = message.author
                    content = message.content
                    message_id = message.id
                    replying_to = message.reference.resolved if message.reference and message.reference.resolved else None
                    msg = Message(author, content, message_id, replying_to)
                    self.logger.info(f"Adding message: {msg.to_dict()}")
                    messages.append(msg)
                    self.logger.info(f"Fetched message: {msg.to_dict()}")
                    self.logger.info(f"Moving on to next message")
            self.logger.info("Done fetching messages")
            self.message_array = messages
            self.message_array.reverse()
            self.logger.info(f"Message history updated. Total messages: {len(self.message_array)}")
        except Exception as e:
            self.logger.error(f"Error in update_message_history: {e}")

    async def send_responses(self, channel):
        while True:
            try:
                ai_response_json = await self.get_ai_response(self.message_array)
                ai_response = ai_response_json["response"]
                picked_message_id = ai_response_json["picked_message"]
                if ai_response and ai_response.strip() not in ["*SILENCE*", "*END OF CONVERSATION*", "", "\n"]:
                    if picked_message_id:
                        picked_message = next((msg for msg in self.message_array if str(msg.message_id) == str(picked_message_id)), None)
                        if picked_message:
                            picked_message.weight *= 0.3  # Lower the weight
                            self.logger.info(f"Picked message: {picked_message.message}")
                            await channel.send(ai_response, reference=discord.MessageReference(message_id=picked_message_id, fail_if_not_exists=False, channel_id=channel.id), mention_author=False)
                        else:
                            self.logger.warning(f"Picked message ID {picked_message_id} not found in message array.")
                            await channel.send(ai_response)
                    else:
                        self.logger.warning("No message picked.")
                        await channel.send(ai_response)
                    self.logger.info(f"Sent response: {ai_response}")
                    if self.DEBUG:
                        debug_context = json.dumps([msg.to_dict() for msg in self.message_array], ensure_ascii=False)
                        debug_msg = f"#[DEBUG - MESSAGE CONTEXT]\n{debug_context}"
                        await channel.send(debug_msg)
                        self.logger.info(f"Debug context: {debug_context}")
                
                if self.NIGHT_MODE_ENABLED and self.is_night_time():
                    delay = random.randint(self.NIGHT_DELAY_MIN, self.NIGHT_DELAY_MAX)
                    self.logger.info(f"Night mode active, delay: {delay}s")
                else:
                    delay = random.randint(self.DELAY_MIN, self.DELAY_MAX)
                    self.logger.info(f"Normal mode, delay: {delay}s")
                
                await self.countdown_status(delay)
            except Exception as e:
                self.logger.error(f"Error in send_responses: {e}")

    def is_night_time(self):
        current_hour = time.localtime().tm_hour
        return 2 <= current_hour < 8

    async def run(self):
        await self.bot.start(self.token)

async def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create bot instances
    v3s_bot = GPTBot({**config['bots']['v3s'], 'openai_key': config['openai_key']})
    sarvel_bot = GPTBot({**config['bots']['sarvel'], 'openai_key': config['openai_key']})
    
    # Run both bots concurrently
    await asyncio.gather(
        v3s_bot.run(),
        sarvel_bot.run()
    )

if __name__ == "__main__":
    asyncio.run(main())
