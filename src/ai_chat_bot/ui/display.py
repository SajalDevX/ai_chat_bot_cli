# src/ai_chatbot/utils/display.py
"""Display utilities for beautiful terminal output."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown


class Display:
    """Handles all terminal display using Rich."""
    
    def __init__(self) -> None:
        """Initialize the display."""
        self.console = Console()
    
    def welcome(self) -> None:
        """Display welcome message at startup."""
        text = Text()
        text.append("🤖 AI Chatbot\n", style="bold cyan")
        text.append("Type your message and press Enter.\n", style="dim")
        text.append("Commands: ", style="dim")
        text.append("/help", style="yellow")
        text.append(", ", style="dim")
        text.append("/clear", style="yellow")
        text.append(", ", style="dim")
        text.append("/quit", style="yellow")
        
        panel = Panel(text, border_style="cyan")
        self.console.print(panel)
        self.console.print()
    
    def user_message(self, content: str) -> None:
        """Display a user's message."""
        self.console.print()
        text = Text()
        text.append("You: ", style="bold green")
        text.append(content)
        self.console.print(text)
    
    def assistant_message(self, content: str) -> None:
        """Display the AI's response (non-streaming)."""
        self.console.print()
        
        header = Text()
        header.append("🤖 Assistant:", style="bold cyan")
        self.console.print(header)
        
        try:
            md = Markdown(content)
            self.console.print(md)
        except Exception:
            self.console.print(content)
    
    # ============ STREAMING METHODS ============
    
    def assistant_stream_start(self) -> None:
        """Start streaming assistant response."""
        self.console.print()
        header = Text()
        header.append("🤖 Assistant:", style="bold cyan")
        self.console.print(header)
        self.console.print()
    
    def assistant_stream_chunk(self, chunk: str) -> None:
        """Print a streaming chunk."""
        self.console.print(chunk, end="", highlight=False)
    
    def assistant_stream_end(self) -> None:
        """End streaming assistant response."""
        self.console.print()
    
    # ============ OTHER METHODS ============
    
    def thinking(self) -> None:
        """Display a 'thinking' indicator."""
        text = Text()
        text.append("🤔 ", style="dim")
        text.append("Thinking...", style="dim italic")
        self.console.print(text)
    
    def stats(self, model: str, tokens: int | None = None) -> None:
        """Display response statistics."""
        self.console.print()
        self.console.print("─" * 50, style="dim")
        
        text = Text()
        text.append("📊 ", style="dim")
        if tokens:
            text.append(f"Tokens: {tokens} | ", style="dim")
        text.append(f"Model: {model}", style="dim")
        self.console.print(text)
    
    def error(self, message: str) -> None:
        """Display an error message."""
        self.console.print()
        panel = Panel(
            Text(message, style="bold red"),
            title="❌ Error",
            border_style="red",
        )
        self.console.print(panel)
    
    def warning(self, message: str) -> None:
        """Display a warning message."""
        self.console.print()
        text = Text()
        text.append("⚠️  ", style="yellow")
        text.append(message, style="yellow")
        self.console.print(text)
    
    def success(self, message: str) -> None:
        """Display a success message."""
        text = Text()
        text.append("✅ ", style="green")
        text.append(message, style="green")
        self.console.print(text)
    
    def info(self, message: str) -> None:
        """Display an info message."""
        text = Text()
        text.append("ℹ️  ", style="blue")
        text.append(message, style="dim")
        self.console.print(text)
    
    def help(self) -> None:
        """Display help information."""
        help_text = """
## Commands

- **/help**  - Show this help message
- **/clear** - Clear conversation history  
- **/quit**  - Exit the chatbot

## Tips

- Just type your message and press Enter
- The AI remembers your conversation history
- Use /clear to start a fresh conversation
"""
        self.console.print()
        md = Markdown(help_text.strip())
        panel = Panel(md, title="📚 Help", border_style="blue")
        self.console.print(panel)
    
    def cleared(self) -> None:
        """Display conversation cleared message."""
        self.success("Conversation cleared. Starting fresh!")
    
    def goodbye(self) -> None:
        """Display goodbye message."""
        self.console.print()
        text = Text()
        text.append("👋 ", style="bold")
        text.append("Goodbye! Thanks for chatting.", style="bold cyan")
        self.console.print(text)
        self.console.print()
    
    def prompt(self) -> str:
        """Display input prompt and get user input."""
        self.console.print()
        try:
            user_input = self.console.input("[bold green]You:[/bold green] ")
            return user_input.strip()
        except EOFError:
            return "/quit"
        except KeyboardInterrupt:
            self.console.print()
            return "/quit"