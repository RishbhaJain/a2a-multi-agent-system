import os
import re
import time
from typing import Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


class WebBrowsingAgent:
    """Agent that can browse the web and interact with websites"""
    
    def __init__(self):
        self.driver = None
        self.wait = None
    
    def _setup_driver(self):
        """Setup Chrome driver with options"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Auto-download and setup ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            return True
        except Exception as e:
            print(f"Error setting up driver: {e}")
            return False
    
    def _cleanup_driver(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    async def invoke(self, prompt: str) -> str:
        """Handle web browsing requests"""
        try:
            if not self._setup_driver():
                return "Error: Could not setup web browser"
            
            # Check if this is a Tic-Tac-Toe request
            if "tic-tac-toe" in prompt.lower() or "ttt.puppy9.com" in prompt.lower():
                return await self._play_tic_tac_toe(prompt)
            else:
                return await self._general_browsing(prompt)
                
        except Exception as e:
            return f"Error in web browsing: {str(e)}"
        finally:
            self._cleanup_driver()
    
    async def _play_tic_tac_toe(self, prompt: str) -> str:
        """Play Tic-Tac-Toe and find the secret number"""
        try:
            # Navigate to the website
            self.driver.get("https://ttt.puppy9.com/")
            time.sleep(2)  # Wait for page to load
            
            # Find the game board (3x3 grid)
            board = self._find_game_board()
            if not board:
                return "Error: Could not find game board"
            
            # Play the game until we win
            game_result = self._play_game_until_win(board)
            if not game_result:
                return "Error: Could not win the game"
            
            # Find the congratulation message and secret number
            secret_number = self._find_secret_number()
            if secret_number:
                return f"Secret number found: {secret_number}"
            else:
                return "Error: Could not find secret number"
                
        except Exception as e:
            return f"Error playing Tic-Tac-Toe: {str(e)}"
    
    def _find_game_board(self):
        """Find the Tic-Tac-Toe game board"""
        try:
            # Based on debug, the game uses buttons with class "cell"
            cells = self.driver.find_elements(By.CSS_SELECTOR, "button.cell")
            if len(cells) >= 9:  # Should have 9 cells for 3x3 grid
                return cells
            return None
        except Exception as e:
            print(f"Error finding board: {e}")
            return None
    
    def _play_game_until_win(self, board):
        """Play Tic-Tac-Toe using optimal strategy"""
        try:
            # Tic-Tac-Toe optimal strategy:
            # 1. Take center if available
            # 2. Take corners
            # 3. Take edges
            
            optimal_moves = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # Center, corners, edges
            
            for i, move in enumerate(optimal_moves):
                if self._make_move(move):
                    time.sleep(2)  # Wait for opponent response
                    
                    # Check if we won
                    if self._check_win():
                        return True
                    
                    # Check if game is over (draw)
                    if self._is_game_over():
                        break
            return False
        except Exception as e:
            print(f"Error playing game: {e}")
            return False
    
    def _make_move(self, position: int):
        """Make a move at the given position"""
        try:
            # Find all game cells (buttons with class "cell")
            cells = self.driver.find_elements(By.CSS_SELECTOR, "button.cell")
            
            if position < len(cells):
                cell = cells[position]
                if cell.is_enabled() and cell.text.strip() == "":
                    cell.click()
                    return True
            return False
        except Exception as e:
            print(f"Error making move: {e}")
            return False
    
    def _check_win(self):
        """Check if we won the game"""
        try:
            # Look for win message
            body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            win_indicators = [
                "you win", "congratulations", "winner", "game won",
                "victory", "success", "you won"
            ]
            
            return any(indicator in body_text for indicator in win_indicators)
        except Exception as e:
            print(f"Error checking win: {e}")
            return False
    
    def _is_game_over(self):
        """Check if the game is over"""
        try:
            # Check if all cells are filled
            cells = self.driver.find_elements(By.CSS_SELECTOR, "button.cell")
            all_filled = all(cell.text.strip() != "" for cell in cells)
            
            # Check for game over message
            body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            game_over_indicators = ["game over", "draw", "tie", "finished"]
            game_over_message = any(indicator in body_text for indicator in game_over_indicators)
            
            return all_filled or game_over_message
        except Exception as e:
            print(f"Error checking game over: {e}")
            return False
    
    def _find_secret_number(self):
        """Find the 14-digit secret number in the page"""
        try:
            # Get page source and look for 14-digit number
            page_text = self.driver.page_source
            
            # Look for 14-digit number pattern
            pattern = r'\b\d{14}\b'
            matches = re.findall(pattern, page_text)
            
            if matches:
                return matches[0]  # Return first 14-digit number found
            
            # Also check visible text
            visible_text = self.driver.find_element(By.TAG_NAME, "body").text
            matches = re.findall(pattern, visible_text)
            
            if matches:
                return matches[0]
            
            return None
        except Exception as e:
            print(f"Error finding secret number: {e}")
            return None
    
    async def _general_browsing(self, prompt: str) -> str:
        """Handle general web browsing requests"""
        try:
            # Extract URL from prompt
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, prompt)
            
            if not urls:
                return "Error: No URL found in the request"
            
            url = urls[0]
            self.driver.get(url)
            time.sleep(2)
            
            # Get page title and some content
            title = self.driver.title
            content = self.driver.find_element(By.TAG_NAME, "body").text[:500]
            
            return f"Page Title: {title}\n\nContent Preview:\n{content}..."
            
        except Exception as e:
            return f"Error browsing website: {str(e)}"
