"""
ULTIMATE KARACHI CRIME SCRAPER - FINAL PRODUCTION VERSION
Features:
- Deep Facebook scraping (multiple techniques)
- Enhanced Twitter scraping with deep crawling
- Reddit scraping with PRAW
- RSS feeds integration
- Google News with deep article crawling
- Pakistani news sites (Dawn, Geo, Tribune, Express)
- AI learning system
- Quality validation
- Multi-threaded for speed
- Robust error handling and recovery
- Execution locking for scheduled runs
"""
import time
import json
import os
import sys
import hashlib
import logging
import logging.handlers
import re
import random
import feedparser
import requests
import traceback
import atexit
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import difflib

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logging.warning("beautifulsoup4 not installed, news scraping will be limited")

# Try to import PRAW for Reddit API
try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False
    logging.warning("praw not installed, Reddit scraping will be disabled")

# Try to import Levenshtein for better performance, fallback to difflib
try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    logging.warning("python-Levenshtein not installed, using difflib for similarity (slower)")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================================
# TASK 1: PROJECT FOUNDATION AND CORE INFRASTRUCTURE
# ============================================================================

class Configuration:
    """Centralized configuration management with environment variable loading"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # OpenRouter API Configuration
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
        self.OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
        
        # Groq API Configuration (fallback for LLM processing)
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
        self.GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
        
        # Gemini API Configuration (fallback #2)
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
        self.GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        # Cerebras API Configuration (fallback #3 - FREE, very fast)
        self.CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY', '')
        self.CEREBRAS_API_KEY2 = os.getenv('CEREBRAS_API_KEY2', '')  # Backup key
        self.CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
        
        # Hugging Face API Configuration (fallback #4 - FREE)
        self.HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY', '')
        self.HUGGING_FACE_MODEL_ID = os.getenv('HUGGING_FACE_MODEL_ID', 'google/gemma-2b-it')
        self.HUGGING_FACE_API_URL = f"https://api-inference.huggingface.co/models/{self.HUGGING_FACE_MODEL_ID}"
        
        # ChatGPT API Configuration (fallback #5 - PAID)
        self.CHAT_GPT_API_KEY = os.getenv('CHAT_GPT_API_KEY', '')
        self.CHAT_GPT_API_URL = "https://api.openai.com/v1/chat/completions"
        
        # DeepSeek API Configuration (fallback #6 - PAID)
        self.DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
        self.DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

        # Test/mocking configuration for local validation
        self.LLM_TEST_MODE = os.getenv('LLM_TEST_MODE', '').strip().lower()
        if self.is_llm_mock_mode():
            self._enable_mock_llm_providers()
        
        # Google Maps API Configuration (for geocoding)
        self.GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
        self.GOOGLE_MAPS_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
        
        # Facebook Graph API Configuration
        self.FB_APP_ID = os.getenv('FB_APP_ID', '')
        self.FB_APP_SECRET = os.getenv('FB_APP_SECRET', '')
        self.FB_GRAPH_API_ACCESS_TOKEN = os.getenv('FB_GRAPH_API_ACCESS_TOKEN', '')
        self.FB_GRAPH_API_URL = "https://graph.facebook.com/v18.0"
        
        # Facebook Login Credentials (for Selenium scraping)
        self.FB_EMAIL = os.getenv('FB_EMAIL', '')
        self.FB_PASSWORD = os.getenv('FB_PASSWORD', '')
        
        # Twitter API Configuration
        self.TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
        self.TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
        self.TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
        self.TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
        self.TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
        
        # Twitter Login Credentials (for Selenium scraping)
        self.TWITTER_USERNAME = os.getenv('TWITTER_USERNAME', '')
        self.TWITTER_PASSWORD = os.getenv('TWITTER_PASSWORD', '')
        
        # Reddit API Configuration
        self.REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
        self.REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'KarachiCrimeScraper/1.0')
        
        # Facebook groups for Karachi crime
        self.FACEBOOK_GROUPS = [
            "https://www.facebook.com/groups/1687885664852771/",
            "https://www.facebook.com/groups/680453496015536/",
            "https://www.facebook.com/groups/6015993931845002/",
            "https://www.facebook.com/groups/karachi1/",
        ]
        
        # Validate critical configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that critical configuration is present"""
        # Count available LLM services
        llm_services = []
        if self.OPENROUTER_API_KEY:
            llm_services.append("OpenRouter")
        if self.GROQ_API_KEY:
            llm_services.append("Groq")
        if self.GEMINI_API_KEY:
            llm_services.append("Gemini")
        if self.CEREBRAS_API_KEY or self.CEREBRAS_API_KEY2:
            llm_services.append("Cerebras")
        if self.HUGGING_FACE_API_KEY:
            llm_services.append("HuggingFace")
        if self.CHAT_GPT_API_KEY:
            llm_services.append("ChatGPT")
        if self.DEEPSEEK_API_KEY:
            llm_services.append("DeepSeek")
        
        if not llm_services:
            logging.warning("No LLM API keys found in .env file - LLM processing will be disabled")
        else:
            logging.info(f"LLM services configured ({len(llm_services)}): {', '.join(llm_services)}")
            logging.info(f"Fallback order: {' → '.join(llm_services)}")
        
        if not self.GOOGLE_MAPS_API_KEY:
            logging.warning("Google Maps API key not found in .env file - Geocoding will be disabled")
        else:
            logging.info("Google Maps API key configured - Geocoding enabled")
        
        if not self.FB_GRAPH_API_ACCESS_TOKEN:
            logging.warning("Facebook Graph API token not found in .env file - Facebook Graph API scraping will be disabled")
        else:
            logging.info("Facebook Graph API configured - Enhanced Facebook scraping enabled")
        
        if not self.TWITTER_BEARER_TOKEN:
            logging.warning("Twitter API credentials not found in .env file - Twitter scraping will be disabled")
        else:
            logging.info("Twitter API configured - Twitter scraping enabled")
        
        if not self.REDDIT_CLIENT_ID or not self.REDDIT_CLIENT_SECRET:
            logging.warning("Reddit API credentials not found in .env file - Reddit scraping will be disabled")

    def _enable_mock_llm_providers(self):
        """
        Enable mock provider keys so tests can run without live credentials.
        This is activated when LLM_TEST_MODE=mock is set in the environment.
        """
        logging.warning("[Configuration] LLM_TEST_MODE=mock - enabling mock provider keys for offline testing")

        def _ensure(key_attr: str, value: str):
            if not getattr(self, key_attr):
                setattr(self, key_attr, value)

        _ensure('OPENROUTER_API_KEY', 'mock-openrouter')
        _ensure('GROQ_API_KEY', 'mock-groq')
        _ensure('GEMINI_API_KEY', 'mock-gemini')
        _ensure('CEREBRAS_API_KEY', 'mock-cerebras')
        _ensure('HUGGING_FACE_API_KEY', 'mock-hf')
        _ensure('CHAT_GPT_API_KEY', 'mock-chatgpt')
        _ensure('DEEPSEEK_API_KEY', 'mock-deepseek')

    def is_llm_mock_mode(self) -> bool:
        """Return True when LLM mock/testing mode is enabled."""
        return self.LLM_TEST_MODE == 'mock'
    
    def has_openrouter_api(self) -> bool:
        """Check if OpenRouter API is configured"""
        return bool(self.OPENROUTER_API_KEY)
    
    def has_groq_api(self) -> bool:
        """Check if Groq API is configured"""
        return bool(self.GROQ_API_KEY)
    
    def has_gemini_api(self) -> bool:
        """Check if Gemini API is configured"""
        return bool(self.GEMINI_API_KEY)
    
    def has_cerebras_api(self) -> bool:
        """Check if Cerebras API is configured"""
        return bool(self.CEREBRAS_API_KEY or self.CEREBRAS_API_KEY2)
    
    def has_huggingface_api(self) -> bool:
        """Check if Hugging Face API is configured"""
        return bool(self.HUGGING_FACE_API_KEY)
    
    def has_chatgpt_api(self) -> bool:
        """Check if ChatGPT API is configured"""
        return bool(self.CHAT_GPT_API_KEY)
    
    def has_deepseek_api(self) -> bool:
        """Check if DeepSeek API is configured"""
        return bool(self.DEEPSEEK_API_KEY)
    
    def has_any_llm_api(self) -> bool:
        """Check if any LLM API is configured"""
        return (self.has_openrouter_api() or self.has_groq_api() or 
                self.has_gemini_api() or self.has_cerebras_api() or 
                self.has_huggingface_api() or self.has_chatgpt_api() or 
                self.has_deepseek_api())
    
    def has_facebook_graph_api(self) -> bool:
        """Check if Facebook Graph API is configured"""
        return bool(self.FB_GRAPH_API_ACCESS_TOKEN)
    
    def has_twitter_api(self) -> bool:
        """Check if Twitter API is configured"""
        return bool(self.TWITTER_BEARER_TOKEN)
    
    def has_reddit_api(self) -> bool:
        """Check if Reddit API is configured"""
        return bool(self.REDDIT_CLIENT_ID and self.REDDIT_CLIENT_SECRET)
    
    def has_google_maps_api(self) -> bool:
        """Check if Google Maps API is configured"""
        return bool(self.GOOGLE_MAPS_API_KEY)
    
    def has_facebook_credentials(self) -> bool:
        """Check if Facebook login credentials are configured"""
        return bool(self.FB_EMAIL and self.FB_PASSWORD)
    
    def has_twitter_credentials(self) -> bool:
        """Check if Twitter login credentials are configured"""
        return bool(self.TWITTER_USERNAME and self.TWITTER_PASSWORD)


class LoggingSystem:
    """Comprehensive logging system with multiple log levels and file handlers"""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files
        self.main_log = self.log_dir / 'scraper.log'
        self.error_log = self.log_dir / 'errors.log'
        self.performance_log = self.log_dir / 'performance.log'
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging with multiple handlers"""
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Formatter for detailed logs
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Formatter for console (simpler)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler (INFO and above) with UTF-8 encoding for Windows
        import io
        # Wrap stdout with UTF-8 encoding to handle Unicode characters on Windows
        if sys.platform == 'win32':
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except AttributeError:
                # Already wrapped or not available
                pass
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Main log file handler (DEBUG and above) with rotation
        main_handler = logging.handlers.RotatingFileHandler(
            self.main_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_handler)
        
        # Error log file handler (ERROR and above) with rotation
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_log,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log file handler (custom level)
        performance_handler = logging.handlers.RotatingFileHandler(
            self.performance_log,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(detailed_formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(performance_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        
        logging.info(f"Logging system initialized - logs directory: {self.log_dir.absolute()}")


class ExecutionLock:
    """Execution lock mechanism to prevent concurrent runs"""
    
    def __init__(self, lock_file: str = '.scraper.lock'):
        self.lock_file = Path(lock_file)
        self.acquired = False
    
    def acquire(self) -> bool:
        """
        Acquire execution lock
        Returns True if lock acquired, False if another instance is running
        """
        if self.lock_file.exists():
            # Check if lock is stale (older than 2 hours)
            try:
                lock_age = time.time() - self.lock_file.stat().st_mtime
                if lock_age > 7200:  # 2 hours
                    logging.warning(f"Removing stale lock file (age: {lock_age/3600:.1f} hours)")
                    self.lock_file.unlink()
                else:
                    logging.error("Another instance is already running (lock file exists)")
                    return False
            except Exception as e:
                logging.error(f"Error checking lock file: {e}")
                return False
        
        try:
            # Create lock file with PID and timestamp
            lock_data = {
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat(),
                'hostname': os.environ.get('COMPUTERNAME', 'unknown')
            }
            self.lock_file.write_text(json.dumps(lock_data, indent=2))
            self.acquired = True
            logging.info(f"Execution lock acquired (PID: {os.getpid()})")
            return True
        except Exception as e:
            logging.error(f"Failed to acquire execution lock: {e}")
            return False
    
    def release(self):
        """Release execution lock"""
        if self.acquired and self.lock_file.exists():
            try:
                self.lock_file.unlink()
                self.acquired = False
                logging.info("Execution lock released")
            except Exception as e:
                logging.error(f"Failed to release execution lock: {e}")
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Cannot acquire execution lock - another instance is running")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class GlobalExceptionHandler:
    """Global exception handler to prevent complete crashes"""
    
    def __init__(self):
        self.original_excepthook = sys.excepthook
        sys.excepthook = self.handle_exception
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        logging.info("Global exception handler installed")
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow KeyboardInterrupt to propagate normally
            logging.info("Execution interrupted by user (Ctrl+C)")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the exception with full traceback
        logging.critical("Uncaught exception occurred:", exc_info=(exc_type, exc_value, exc_traceback))
        logging.critical("="*80)
        logging.critical("CRITICAL ERROR - Scraper encountered an unhandled exception")
        logging.critical("="*80)
        
        # Format traceback for logging
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in tb_lines:
            logging.critical(line.rstrip())
        
        logging.critical("="*80)
        logging.critical("The scraper will attempt to save any collected data before exiting")
        logging.critical("="*80)
        
        # Call original exception hook for system handling
        self.original_excepthook(exc_type, exc_value, exc_traceback)
    
    def cleanup(self):
        """Cleanup on exit"""
        logging.debug("Global exception handler cleanup")


# Initialize global components
config = Configuration()
logging_system = LoggingSystem()
exception_handler = GlobalExceptionHandler()
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 3: DRIVER POOL MANAGER WITH HEALTH MONITORING
# ============================================================================

import threading
from enum import Enum
from typing import Callable, Any


class DriverPoolManager:
    """
    Manages a pool of WebDriver instances with health monitoring and automatic recovery.
    Implements connection pooling, health checks, and automatic driver restart on failures.
    
    Features:
    - Connection pooling with configurable pool size
    - Anti-detection measures (headless, user agent rotation)
    - Health check mechanism for driver responsiveness
    - Safe execution wrapper with automatic recovery
    - Driver restart logic with session restoration
    - Performance tracking (response time, request count)
    """
    
    def __init__(self, pool_size: int = 3):
        """
        Initialize driver pool manager
        
        Args:
            pool_size: Number of drivers to maintain in the pool (default: 3)
        """
        self.pool_size = pool_size
        self.pool: List[webdriver.Chrome] = []
        self.health: Dict[str, DriverHealth] = {}
        self.lock = threading.Lock()
        self.driver_counter = 0
        
        # User agents for rotation (anti-detection)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'
        ]
        
        logging.info(f"[DriverPool] Initialized with pool size: {pool_size}")
    
    def _create_driver(self) -> webdriver.Chrome:
        """
        Create a new WebDriver instance with anti-detection measures
        
        Returns:
            Configured Chrome WebDriver instance
        """
        options = Options()
        
        # Anti-detection measures
        options.add_argument('--headless=new')  # Run in new headless mode (more stable)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-cache')
        options.add_argument('--disk-cache-size=1')
        
        # SSL error handling - ignore certificate errors
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--allow-insecure-localhost')
        options.add_argument('--ignore-certificate-errors-spki-list')
        
        # Memory and stability improvements to prevent tab crashes
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-background-networking')
        options.add_argument('--disable-background-timer-throttling')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-breakpad')
        options.add_argument('--disable-component-extensions-with-background-pages')
        options.add_argument('--disable-features=TranslateUI,BlinkGenPropertyTrees')
        options.add_argument('--disable-ipc-flooding-protection')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--enable-features=NetworkService,NetworkServiceInProcess')
        options.add_argument('--force-color-profile=srgb')
        options.add_argument('--hide-scrollbars')
        options.add_argument('--metrics-recording-only')
        options.add_argument('--mute-audio')
        options.add_argument('--window-size=1920,1080')
        
        # Reduce memory usage
        options.add_argument('--single-process')
        options.add_argument('--disable-web-security')
        options.add_argument('--aggressive-cache-discard')
        options.add_argument('--disable-application-cache')
        
        # Rotate user agent for anti-detection
        user_agent = random.choice(self.user_agents)
        options.add_argument(f'user-agent={user_agent}')
        
        # Exclude automation flags
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Set page load strategy
        options.page_load_strategy = 'normal'
        
        try:
            # Create driver
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            
            # Execute script to hide webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set timeouts
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            # Generate unique driver ID
            self.driver_counter += 1
            driver_id = f"driver_{self.driver_counter}_{id(driver)}"
            
            # Initialize health tracking
            self.health[driver_id] = DriverHealth(
                driver_id=driver_id,
                is_alive=True,
                last_check=datetime.now().isoformat(),
                consecutive_failures=0,
                total_requests=0,
                avg_response_time=0.0
            )
            
            # Store driver ID as attribute
            driver.driver_id = driver_id
            
            logging.info(f"[DriverPool] Created new driver: {driver_id}")
            return driver
            
        except Exception as e:
            logging.error(f"[DriverPool] Failed to create driver: {e}")
            raise
    
    def get_driver(self) -> webdriver.Chrome:
        """
        Get a healthy driver from the pool
        Creates new driver if pool is empty or all drivers are unhealthy
        
        Returns:
            Healthy WebDriver instance
        """
        with self.lock:
            # Try to find a healthy driver in the pool
            for driver in self.pool:
                if self.check_health(driver):
                    self.pool.remove(driver)
                    logging.debug(f"[DriverPool] Retrieved driver from pool: {driver.driver_id}")
                    return driver
            
            # No healthy drivers available, create new one
            if len(self.pool) < self.pool_size:
                driver = self._create_driver()
                logging.info(f"[DriverPool] Created new driver (pool size: {len(self.pool) + 1}/{self.pool_size})")
                return driver
            else:
                # Pool is full, restart oldest driver
                old_driver = self.pool.pop(0)
                logging.warning(f"[DriverPool] Pool full, restarting oldest driver: {old_driver.driver_id}")
                return self.restart_driver(old_driver)
    
    def release_driver(self, driver: webdriver.Chrome):
        """
        Return driver to the pool for reuse
        
        Args:
            driver: WebDriver instance to return to pool
        """
        with self.lock:
            if driver and hasattr(driver, 'driver_id'):
                # Check health before returning to pool
                if self.check_health(driver):
                    if len(self.pool) < self.pool_size:
                        self.pool.append(driver)
                        logging.debug(f"[DriverPool] Returned driver to pool: {driver.driver_id}")
                    else:
                        # Pool is full, close this driver
                        self._close_driver(driver)
                        logging.debug(f"[DriverPool] Pool full, closed driver: {driver.driver_id}")
                else:
                    # Driver is unhealthy, close it
                    self._close_driver(driver)
                    logging.warning(f"[DriverPool] Driver unhealthy, closed: {driver.driver_id}")
    
    def check_health(self, driver: webdriver.Chrome) -> bool:
        """
        Verify driver is responsive and healthy
        
        Args:
            driver: WebDriver instance to check
            
        Returns:
            True if driver is healthy, False otherwise
        """
        if not driver or not hasattr(driver, 'driver_id'):
            return False
        
        driver_id = driver.driver_id
        start_time = time.time()
        
        try:
            # Try to get current URL (simple health check)
            _ = driver.current_url
            
            # Try to execute a simple script
            driver.execute_script("return document.readyState;")
            
            # Update health status
            response_time = time.time() - start_time
            
            if driver_id in self.health:
                health = self.health[driver_id]
                health.is_alive = True
                health.last_check = datetime.now().isoformat()
                health.consecutive_failures = 0
                
                # Update average response time
                total_requests = health.total_requests
                health.avg_response_time = (
                    (health.avg_response_time * total_requests + response_time) / 
                    (total_requests + 1)
                )
                health.total_requests += 1
                
                # Log health check every 50 operations (as per requirement)
                if health.total_requests % 50 == 0:
                    logging.info(
                        f"[DriverPool] Health check {driver_id}: "
                        f"requests={health.total_requests}, "
                        f"avg_response={health.avg_response_time:.3f}s"
                    )
            
            return True
            
        except Exception as e:
            logging.warning(f"[DriverPool] Health check failed for {driver_id}: {e}")
            
            # Update health status
            if driver_id in self.health:
                health = self.health[driver_id]
                health.is_alive = False
                health.last_check = datetime.now().isoformat()
                health.consecutive_failures += 1
            
            return False
    
    def restart_driver(self, driver: webdriver.Chrome) -> webdriver.Chrome:
        """
        Restart a crashed driver with session restoration
        
        Args:
            driver: Failed WebDriver instance
            
        Returns:
            New WebDriver instance
        """
        old_driver_id = driver.driver_id if hasattr(driver, 'driver_id') else 'unknown'
        
        logging.info(f"[DriverPool] Restarting driver: {old_driver_id}")
        
        # Save current URL for session restoration (if possible)
        current_url = None
        try:
            current_url = driver.current_url
        except:
            pass
        
        # Close old driver
        self._close_driver(driver)
        
        # Create new driver
        try:
            new_driver = self._create_driver()
            
            # Attempt session restoration
            if current_url and current_url != 'data:,':
                try:
                    new_driver.get(current_url)
                    logging.info(f"[DriverPool] Session restored to: {current_url}")
                except Exception as e:
                    logging.warning(f"[DriverPool] Could not restore session: {e}")
            
            logging.info(f"[DriverPool] Driver restarted successfully: {new_driver.driver_id}")
            return new_driver
            
        except Exception as e:
            logging.error(f"[DriverPool] Failed to restart driver: {e}")
            raise
    
    def safe_execute(self, driver: webdriver.Chrome, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with automatic recovery on failure
        Wraps operations with try-catch and automatic driver restart
        
        Args:
            driver: WebDriver instance to use
            operation: Callable operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
            
        Raises:
            Exception if operation fails after retries
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check driver health before operation
                if not self.check_health(driver):
                    logging.warning(f"[DriverPool] Driver unhealthy before operation, restarting...")
                    driver = self.restart_driver(driver)
                
                # Execute operation
                start_time = time.time()
                result = operation(driver, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update performance tracking
                if hasattr(driver, 'driver_id') and driver.driver_id in self.health:
                    health = self.health[driver.driver_id]
                    health.total_requests += 1
                    
                    # Update average response time
                    total = health.total_requests
                    health.avg_response_time = (
                        (health.avg_response_time * (total - 1) + execution_time) / total
                    )
                
                logging.debug(f"[DriverPool] Operation completed in {execution_time:.2f}s")
                return result
                
            except WebDriverException as e:
                retry_count += 1
                error_msg = str(e)
                
                logging.error(
                    f"[DriverPool] WebDriver error (attempt {retry_count}/{max_retries}): {error_msg[:100]}"
                )
                
                # Update failure count
                if hasattr(driver, 'driver_id') and driver.driver_id in self.health:
                    self.health[driver.driver_id].consecutive_failures += 1
                
                if retry_count < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    logging.info(f"[DriverPool] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
                    # Restart driver for next attempt
                    try:
                        driver = self.restart_driver(driver)
                    except Exception as restart_error:
                        logging.error(f"[DriverPool] Failed to restart driver: {restart_error}")
                        if retry_count >= max_retries:
                            raise
                else:
                    logging.error(f"[DriverPool] Max retries reached, operation failed")
                    raise
                    
            except Exception as e:
                logging.error(f"[DriverPool] Unexpected error in safe_execute: {e}")
                raise
        
        raise RuntimeError(f"Operation failed after {max_retries} retries")
    
    def _close_driver(self, driver: webdriver.Chrome):
        """
        Safely close a driver instance
        
        Args:
            driver: WebDriver instance to close
        """
        if not driver:
            return
        
        driver_id = driver.driver_id if hasattr(driver, 'driver_id') else 'unknown'
        
        try:
            driver.quit()
            logging.debug(f"[DriverPool] Closed driver: {driver_id}")
        except Exception as e:
            logging.warning(f"[DriverPool] Error closing driver {driver_id}: {e}")
        finally:
            # Mark as dead in health tracking
            if driver_id in self.health:
                self.health[driver_id].is_alive = False
    
    def close_all(self):
        """Close all drivers in the pool"""
        with self.lock:
            logging.info(f"[DriverPool] Closing all drivers (count: {len(self.pool)})")
            
            for driver in self.pool:
                self._close_driver(driver)
            
            self.pool.clear()
            logging.info("[DriverPool] All drivers closed")
    
    def get_pool_stats(self) -> Dict:
        """
        Get statistics about the driver pool
        
        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            stats = {
                'pool_size': len(self.pool),
                'max_pool_size': self.pool_size,
                'total_drivers_created': self.driver_counter,
                'drivers': []
            }
            
            for driver_id, health in self.health.items():
                stats['drivers'].append({
                    'driver_id': driver_id,
                    'is_alive': health.is_alive,
                    'total_requests': health.total_requests,
                    'avg_response_time': health.avg_response_time,
                    'consecutive_failures': health.consecutive_failures,
                    'last_check': health.last_check
                })
            
            return stats
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all drivers"""
        self.close_all()
        return False


# ============================================================================
# TASK 4: COMPREHENSIVE ERROR RECOVERY SYSTEM
# ============================================================================

class ErrorType(Enum):
    """Classification of error types for targeted recovery strategies"""
    DRIVER_CRASH = "driver_crash"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    SSL_ERROR = "ssl_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PARSING_ERROR = "parsing_error"
    API_ERROR = "api_error"
    FILE_IO_ERROR = "file_io_error"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions to take for different error types"""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RESTART_DRIVER = "restart_driver"
    SKIP_AND_CONTINUE = "skip_and_continue"
    SWITCH_METHOD = "switch_method"
    HALT_AND_REPORT = "halt_and_report"
    DEGRADE_GRACEFULLY = "degrade_gracefully"


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing if service recovered
    
    Features:
    - Automatic state transitions based on failure threshold
    - Timeout-based recovery attempts
    - Success tracking to close circuit
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, success_threshold: int = 2):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery (OPEN -> HALF_OPEN)
            success_threshold: Number of successes needed to close circuit from HALF_OPEN
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        self.lock = threading.Lock()
        
        logging.info(f"[CircuitBreaker] Initialized (threshold={failure_threshold}, timeout={timeout}s)")
    
    def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation through circuit breaker
        
        Args:
            operation: Callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
            
        Raises:
            RuntimeError: If circuit is OPEN
            Exception: If operation fails
        """
        with self.lock:
            # Check if we should attempt recovery
            if self.state == "OPEN":
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.timeout:
                        logging.info(f"[CircuitBreaker] Attempting recovery (OPEN -> HALF_OPEN)")
                        self.state = "HALF_OPEN"
                        self.success_count = 0
                    else:
                        raise RuntimeError(
                            f"Circuit breaker is OPEN. "
                            f"Retry in {self.timeout - elapsed:.0f}s"
                        )
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
        
        # Execute operation
        try:
            result = operation(*args, **kwargs)
            self.record_success()
            return result
            
        except Exception as e:
            self.record_failure()
            raise
    
    def record_success(self):
        """Record successful operation"""
        with self.lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                logging.debug(
                    f"[CircuitBreaker] Success in HALF_OPEN "
                    f"({self.success_count}/{self.success_threshold})"
                )
                
                if self.success_count >= self.success_threshold:
                    logging.info(f"[CircuitBreaker] Circuit closed (HALF_OPEN -> CLOSED)")
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
                    self.last_failure_time = None
            
            elif self.state == "CLOSED":
                # Reset failure count on success
                if self.failure_count > 0:
                    logging.debug(f"[CircuitBreaker] Resetting failure count")
                    self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "HALF_OPEN":
                logging.warning(f"[CircuitBreaker] Failure in HALF_OPEN (HALF_OPEN -> OPEN)")
                self.state = "OPEN"
                self.success_count = 0
            
            elif self.state == "CLOSED":
                logging.warning(
                    f"[CircuitBreaker] Failure recorded "
                    f"({self.failure_count}/{self.failure_threshold})"
                )
                
                if self.failure_count >= self.failure_threshold:
                    logging.error(f"[CircuitBreaker] Threshold reached (CLOSED -> OPEN)")
                    self.state = "OPEN"
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        with self.lock:
            logging.info(f"[CircuitBreaker] Manual reset to CLOSED")
            self.state = "CLOSED"
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
    
    def get_state(self) -> Dict:
        """Get current circuit breaker state"""
        with self.lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time
            }


class ErrorRecoverySystem:
    """
    Comprehensive error recovery system with intelligent error classification
    and recovery strategy routing.
    
    Features:
    - Automatic error type detection and classification
    - Recovery strategy routing based on error type
    - Exponential backoff retry logic
    - Circuit breaker integration
    - Error pattern detection for automatic halting
    - Detailed error logging with context
    
    Requirements addressed: 1, 2, 3, 13, 24, 25, 30
    """
    
    def __init__(self, driver_pool: Optional[DriverPoolManager] = None):
        """
        Initialize error recovery system
        
        Args:
            driver_pool: Optional DriverPoolManager for driver restart operations
        """
        self.driver_pool = driver_pool
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[Dict] = []
        self.error_patterns: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        # Error pattern thresholds for automatic halting
        self.halt_thresholds = {
            ErrorType.DRIVER_CRASH: 10,
            ErrorType.CONNECTION_ERROR: 15,
            ErrorType.SSL_ERROR: 10,
            ErrorType.AUTHENTICATION: 5,
            ErrorType.API_ERROR: 10
        }
        
        logging.info("[ErrorRecovery] System initialized")
    
    def classify_error(self, error: Exception, context: Dict = None) -> ErrorType:
        """
        Classify error into appropriate error type
        
        Args:
            error: Exception to classify
            context: Optional context dictionary with additional information
            
        Returns:
            ErrorType enum value
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Driver crash detection (WinError 10061, connection refused)
        if 'winerror 10061' in error_str or 'connection refused' in error_str:
            return ErrorType.DRIVER_CRASH
        
        if isinstance(error, WebDriverException):
            if 'timeout' in error_str or 'timed out' in error_str:
                return ErrorType.TIMEOUT
            elif 'no such element' in error_str or 'stale element' in error_str:
                return ErrorType.PARSING_ERROR
            else:
                return ErrorType.DRIVER_CRASH
        
        # Connection errors
        if 'connectionerror' in error_type_name or 'connection' in error_str:
            return ErrorType.CONNECTION_ERROR
        
        # Timeout errors
        if 'timeout' in error_type_name or 'timeout' in error_str:
            return ErrorType.TIMEOUT
        
        # SSL/TLS errors
        if 'ssl' in error_str or 'certificate' in error_str or 'tls' in error_str:
            return ErrorType.SSL_ERROR
        
        # Rate limiting
        if '429' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            return ErrorType.RATE_LIMIT
        
        # Authentication errors
        if '401' in error_str or '403' in error_str or 'unauthorized' in error_str or 'forbidden' in error_str:
            return ErrorType.AUTHENTICATION
        
        # API errors
        if 'api' in error_str or '500' in error_str or '502' in error_str or '503' in error_str:
            return ErrorType.API_ERROR
        
        # File I/O errors
        if 'ioerror' in error_type_name or 'file' in error_str or 'permission' in error_str:
            return ErrorType.FILE_IO_ERROR
        
        return ErrorType.UNKNOWN
    
    def get_recovery_action(self, error_type: ErrorType, context: Dict = None) -> RecoveryAction:
        """
        Determine appropriate recovery action for error type
        
        Args:
            error_type: Classified error type
            context: Optional context dictionary
            
        Returns:
            RecoveryAction enum value
        """
        # Recovery strategy mapping
        recovery_map = {
            ErrorType.DRIVER_CRASH: RecoveryAction.RESTART_DRIVER,
            ErrorType.CONNECTION_ERROR: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorType.TIMEOUT: RecoveryAction.RETRY_IMMEDIATE,
            ErrorType.SSL_ERROR: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorType.RATE_LIMIT: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorType.AUTHENTICATION: RecoveryAction.SWITCH_METHOD,
            ErrorType.PARSING_ERROR: RecoveryAction.SKIP_AND_CONTINUE,
            ErrorType.API_ERROR: RecoveryAction.RETRY_WITH_BACKOFF,
            ErrorType.FILE_IO_ERROR: RecoveryAction.HALT_AND_REPORT,
            ErrorType.UNKNOWN: RecoveryAction.SKIP_AND_CONTINUE
        }
        
        return recovery_map.get(error_type, RecoveryAction.SKIP_AND_CONTINUE)
    
    def handle_driver_error(self, error: Exception, context: Dict = None) -> RecoveryAction:
        """
        Handle driver-specific errors (WinError 10061, crashes)
        
        Args:
            error: Driver exception
            context: Context with driver information
            
        Returns:
            RecoveryAction to take
        """
        logging.error(f"[ErrorRecovery] Driver error detected: {error}")
        
        # Log context
        if context:
            logging.error(f"[ErrorRecovery] Context: {json.dumps(context, indent=2)}")
        
        # Record error pattern
        self._record_error_pattern(ErrorType.DRIVER_CRASH)
        
        # Check if we should halt
        if self._should_halt(ErrorType.DRIVER_CRASH):
            logging.critical("[ErrorRecovery] Too many driver crashes - halting execution")
            return RecoveryAction.HALT_AND_REPORT
        
        return RecoveryAction.RESTART_DRIVER
    
    def handle_network_error(self, error: Exception, context: Dict = None) -> RecoveryAction:
        """
        Handle network and connection errors
        
        Args:
            error: Network exception
            context: Context with request information
            
        Returns:
            RecoveryAction to take
        """
        logging.warning(f"[ErrorRecovery] Network error: {error}")
        
        # Record error pattern
        self._record_error_pattern(ErrorType.CONNECTION_ERROR)
        
        # Check if we should halt
        if self._should_halt(ErrorType.CONNECTION_ERROR):
            logging.critical("[ErrorRecovery] Too many connection errors - halting execution")
            return RecoveryAction.HALT_AND_REPORT
        
        return RecoveryAction.RETRY_WITH_BACKOFF
    
    def handle_ssl_error(self, error: Exception, context: Dict = None) -> RecoveryAction:
        """
        Handle SSL/TLS certificate errors
        
        Args:
            error: SSL exception
            context: Context with URL information
            
        Returns:
            RecoveryAction to take
        """
        logging.warning(f"[ErrorRecovery] SSL error: {error}")
        
        if context and 'url' in context:
            logging.warning(f"[ErrorRecovery] Problematic URL: {context['url']}")
        
        # Record error pattern
        self._record_error_pattern(ErrorType.SSL_ERROR)
        
        # Check if we should halt
        if self._should_halt(ErrorType.SSL_ERROR):
            logging.critical("[ErrorRecovery] Too many SSL errors - halting execution")
            return RecoveryAction.HALT_AND_REPORT
        
        return RecoveryAction.RETRY_WITH_BACKOFF
    
    def handle_rate_limit(self, error: Exception, context: Dict = None) -> RecoveryAction:
        """
        Handle rate limiting from APIs and websites
        
        Args:
            error: Rate limit exception
            context: Context with source information
            
        Returns:
            RecoveryAction to take
        """
        logging.warning(f"[ErrorRecovery] Rate limit hit: {error}")
        
        if context and 'source' in context:
            logging.warning(f"[ErrorRecovery] Source: {context['source']}")
        
        return RecoveryAction.RETRY_WITH_BACKOFF
    
    def handle_api_error(self, error: Exception, context: Dict = None) -> RecoveryAction:
        """
        Handle API errors (OpenRouter, Reddit, etc.)
        
        Args:
            error: API exception
            context: Context with API information
            
        Returns:
            RecoveryAction to take
        """
        logging.warning(f"[ErrorRecovery] API error: {error}")
        
        # Record error pattern
        self._record_error_pattern(ErrorType.API_ERROR)
        
        # Check if we should halt
        if self._should_halt(ErrorType.API_ERROR):
            logging.critical("[ErrorRecovery] Too many API errors - halting execution")
            return RecoveryAction.HALT_AND_REPORT
        
        return RecoveryAction.RETRY_WITH_BACKOFF
    
    def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        initial_delay: float = 2.0,
        backoff_factor: float = 2.0,
        context: Dict = None,
        circuit_breaker_key: str = None
    ) -> Any:
        """
        Execute operation with exponential backoff retry logic
        
        Args:
            operation: Callable to execute
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            backoff_factor: Multiplier for exponential backoff
            context: Optional context dictionary for error logging
            circuit_breaker_key: Optional key for circuit breaker (creates one if not exists)
            
        Returns:
            Result of operation
            
        Raises:
            Exception: If operation fails after all retries
        """
        retry_count = 0
        delay = initial_delay
        last_error = None
        
        # Get or create circuit breaker if key provided
        circuit_breaker = None
        if circuit_breaker_key:
            circuit_breaker = self._get_circuit_breaker(circuit_breaker_key)
        
        while retry_count <= max_retries:
            try:
                # Execute through circuit breaker if available
                if circuit_breaker:
                    result = circuit_breaker.call(operation)
                else:
                    result = operation()
                
                # Success - log if this was a retry
                if retry_count > 0:
                    logging.info(
                        f"[ErrorRecovery] Operation succeeded after {retry_count} retries"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Classify error
                error_type = self.classify_error(e, context)
                
                # Log error with context
                self._log_error(e, error_type, context, retry_count, max_retries)
                
                # Record in error history
                self._record_error(e, error_type, context)
                
                # Check if we've exhausted retries
                if retry_count > max_retries:
                    logging.error(
                        f"[ErrorRecovery] Max retries ({max_retries}) exceeded for operation"
                    )
                    break
                
                # Get recovery action
                recovery_action = self.get_recovery_action(error_type, context)
                
                # Handle specific recovery actions
                if recovery_action == RecoveryAction.HALT_AND_REPORT:
                    logging.critical("[ErrorRecovery] Halting execution due to critical error")
                    raise
                
                elif recovery_action == RecoveryAction.SKIP_AND_CONTINUE:
                    logging.warning("[ErrorRecovery] Skipping operation and continuing")
                    return None
                
                elif recovery_action == RecoveryAction.RESTART_DRIVER:
                    if self.driver_pool and context and 'driver' in context:
                        logging.info("[ErrorRecovery] Restarting driver...")
                        try:
                            context['driver'] = self.driver_pool.restart_driver(context['driver'])
                        except Exception as restart_error:
                            logging.error(f"[ErrorRecovery] Driver restart failed: {restart_error}")
                
                # Apply exponential backoff
                if recovery_action in [RecoveryAction.RETRY_WITH_BACKOFF, RecoveryAction.RETRY_IMMEDIATE]:
                    if recovery_action == RecoveryAction.RETRY_WITH_BACKOFF:
                        # Special handling for rate limits
                        if error_type == ErrorType.RATE_LIMIT:
                            delay = 60.0  # Wait 60 seconds for rate limits
                        
                        logging.info(f"[ErrorRecovery] Waiting {delay:.1f}s before retry {retry_count}/{max_retries}")
                        time.sleep(delay)
                        delay *= backoff_factor  # Exponential backoff
                    else:
                        # Immediate retry with small delay
                        time.sleep(0.5)
        
        # All retries exhausted
        if last_error:
            logging.error(f"[ErrorRecovery] Operation failed after {max_retries} retries")
            raise last_error
        
        raise RuntimeError(f"Operation failed after {max_retries} retries")
    
    def _get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for key"""
        with self.lock:
            if key not in self.circuit_breakers:
                self.circuit_breakers[key] = CircuitBreaker()
                logging.debug(f"[ErrorRecovery] Created circuit breaker: {key}")
            return self.circuit_breakers[key]
    
    def _record_error(self, error: Exception, error_type: ErrorType, context: Dict = None):
        """Record error in history"""
        with self.lock:
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error_type.value,
                'error_message': str(error),
                'error_class': type(error).__name__,
                'context': context or {}
            }
            
            self.error_history.append(error_record)
            
            # Keep only last 100 errors
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]
    
    def _record_error_pattern(self, error_type: ErrorType):
        """Record error pattern for automatic halting detection"""
        with self.lock:
            key = error_type.value
            self.error_patterns[key] = self.error_patterns.get(key, 0) + 1
    
    def _should_halt(self, error_type: ErrorType) -> bool:
        """Check if we should halt execution based on error patterns"""
        with self.lock:
            key = error_type.value
            count = self.error_patterns.get(key, 0)
            threshold = self.halt_thresholds.get(error_type, 20)
            
            return count >= threshold
    
    def _log_error(
        self,
        error: Exception,
        error_type: ErrorType,
        context: Dict,
        retry_count: int,
        max_retries: int
    ):
        """Log error with full context"""
        logging.error("="*80)
        logging.error(f"[ErrorRecovery] Error occurred (attempt {retry_count}/{max_retries})")
        logging.error(f"[ErrorRecovery] Error Type: {error_type.value}")
        logging.error(f"[ErrorRecovery] Error Class: {type(error).__name__}")
        logging.error(f"[ErrorRecovery] Error Message: {error}")
        
        if context:
            logging.error(f"[ErrorRecovery] Context:")
            for key, value in context.items():
                if key != 'driver':  # Don't log driver object
                    logging.error(f"  - {key}: {value}")
        
        # Log stack trace for critical errors
        if error_type in [ErrorType.DRIVER_CRASH, ErrorType.FILE_IO_ERROR]:
            logging.error(f"[ErrorRecovery] Stack trace:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logging.error(f"  {line}")
        
        logging.error("="*80)
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics for monitoring"""
        with self.lock:
            return {
                'total_errors': len(self.error_history),
                'error_patterns': dict(self.error_patterns),
                'circuit_breakers': {
                    key: cb.get_state()
                    for key, cb in self.circuit_breakers.items()
                },
                'recent_errors': self.error_history[-10:] if self.error_history else []
            }
    
    def reset_error_patterns(self):
        """Reset error pattern counters (useful for new runs)"""
        with self.lock:
            self.error_patterns.clear()
            logging.info("[ErrorRecovery] Error patterns reset")
    
    def generate_diagnostic_report(self) -> str:
        """Generate diagnostic report for troubleshooting"""
        stats = self.get_error_statistics()
        
        report = []
        report.append("="*80)
        report.append("ERROR RECOVERY DIAGNOSTIC REPORT")
        report.append("="*80)
        report.append(f"Total Errors Recorded: {stats['total_errors']}")
        report.append("")
        
        report.append("Error Patterns:")
        for error_type, count in stats['error_patterns'].items():
            threshold = self.halt_thresholds.get(ErrorType(error_type), 20)
            report.append(f"  - {error_type}: {count}/{threshold}")
        report.append("")
        
        report.append("Circuit Breakers:")
        for key, state in stats['circuit_breakers'].items():
            report.append(f"  - {key}: {state['state']} (failures: {state['failure_count']})")
        report.append("")
        
        if stats['recent_errors']:
            report.append("Recent Errors (last 10):")
            for err in stats['recent_errors']:
                report.append(f"  - [{err['timestamp']}] {err['error_type']}: {err['error_message'][:100]}")
        
        report.append("="*80)
        
        return '\n'.join(report)


# ============================================================================
# TASK 5: HTTP CLIENT POOL WITH RETRY LOGIC
# ============================================================================

import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as Urllib3Retry


class HTTPClientPool:
    """
    HTTP client pool with connection pooling, retry logic, and anti-detection measures.
    
    Features:
    - Connection pooling for improved performance
    - Exponential backoff retry logic for network errors
    - SSL error handling with certificate verification fallback
    - Rate limiting and request throttling
    - Configurable request timeout (30 seconds default)
    - User agent rotation for anti-detection
    - Circuit breaker integration
    
    Requirements addressed: 3, 11, 13, 24, 27
    """
    
    def __init__(
        self,
        pool_size: int = 10,
        max_retries: int = 3,
        timeout: int = 30,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize HTTP client pool
        
        Args:
            pool_size: Maximum number of connections in pool (default: 10)
            max_retries: Maximum retry attempts for failed requests (default: 3)
            timeout: Request timeout in seconds (default: 30)
            rate_limit_delay: Minimum delay between requests in seconds (default: 0.5)
        """
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        
        # User agents for rotation (anti-detection)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        ]
        
        # Session pool for connection reuse
        self.sessions: Dict[str, requests.Session] = {}
        self.session_lock = threading.Lock()
        
        # Rate limiting tracking
        self.last_request_time: Dict[str, float] = {}
        self.rate_limit_lock = threading.Lock()
        
        # Request statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'ssl_errors': 0,
            'timeout_errors': 0,
            'connection_errors': 0
        }
        self.stats_lock = threading.Lock()
        
        logging.info(
            f"[HTTPClientPool] Initialized (pool_size={pool_size}, "
            f"max_retries={max_retries}, timeout={timeout}s)"
        )
    
    def _get_session(self, domain: str = 'default') -> requests.Session:
        """
        Get or create a session for the given domain
        Sessions are reused for connection pooling
        
        Args:
            domain: Domain name for session isolation (default: 'default')
            
        Returns:
            Configured requests.Session instance
        """
        with self.session_lock:
            if domain not in self.sessions:
                session = requests.Session()
                
                # Configure retry strategy with exponential backoff
                retry_strategy = Urllib3Retry(
                    total=self.max_retries,
                    backoff_factor=2,  # Exponential backoff: 2, 4, 8 seconds
                    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Methods to retry
                    raise_on_status=False
                )
                
                # Create adapter with retry strategy and connection pooling
                adapter = HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=self.pool_size,
                    pool_maxsize=self.pool_size,
                    pool_block=False
                )
                
                # Mount adapter for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # Set default headers
                session.headers.update({
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                })
                
                self.sessions[domain] = session
                logging.debug(f"[HTTPClientPool] Created new session for domain: {domain}")
            
            return self.sessions[domain]
    
    def _get_random_user_agent(self) -> str:
        """
        Get a random user agent for anti-detection
        
        Returns:
            Random user agent string
        """
        return random.choice(self.user_agents)
    
    def _apply_rate_limit(self, domain: str):
        """
        Apply rate limiting delay for the given domain
        
        Args:
            domain: Domain name for rate limiting
        """
        with self.rate_limit_lock:
            if domain in self.last_request_time:
                elapsed = time.time() - self.last_request_time[domain]
                if elapsed < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - elapsed
                    logging.debug(f"[HTTPClientPool] Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                    time.sleep(sleep_time)
            
            self.last_request_time[domain] = time.time()
    
    def _update_stats(self, stat_key: str):
        """Update request statistics"""
        with self.stats_lock:
            self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL for session management
        
        Args:
            url: Full URL
            
        Returns:
            Domain name
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or 'default'
        except:
            return 'default'
    
    def get(
        self,
        url: str,
        headers: Dict = None,
        params: Dict = None,
        timeout: int = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        retry_on_ssl_error: bool = True
    ) -> Optional[requests.Response]:
        """
        Perform GET request with retry logic and error handling
        
        Args:
            url: URL to request
            headers: Optional custom headers
            params: Optional query parameters
            timeout: Optional custom timeout (uses default if not specified)
            verify_ssl: Whether to verify SSL certificates (default: True)
            allow_redirects: Whether to follow redirects (default: True)
            retry_on_ssl_error: Whether to retry with SSL verification disabled on SSL errors
            
        Returns:
            Response object or None if request failed
        """
        return self._request(
            method='GET',
            url=url,
            headers=headers,
            params=params,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            retry_on_ssl_error=retry_on_ssl_error
        )
    
    def post(
        self,
        url: str,
        data: Dict = None,
        json_data: Dict = None,
        headers: Dict = None,
        timeout: int = None,
        verify_ssl: bool = True,
        retry_on_ssl_error: bool = True
    ) -> Optional[requests.Response]:
        """
        Perform POST request with retry logic and error handling
        
        Args:
            url: URL to request
            data: Optional form data
            json_data: Optional JSON data
            headers: Optional custom headers
            timeout: Optional custom timeout (uses default if not specified)
            verify_ssl: Whether to verify SSL certificates (default: True)
            retry_on_ssl_error: Whether to retry with SSL verification disabled on SSL errors
            
        Returns:
            Response object or None if request failed
        """
        return self._request(
            method='POST',
            url=url,
            data=data,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            verify_ssl=verify_ssl,
            retry_on_ssl_error=retry_on_ssl_error
        )
    
    def _request(
        self,
        method: str,
        url: str,
        headers: Dict = None,
        params: Dict = None,
        data: Dict = None,
        json_data: Dict = None,
        timeout: int = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        retry_on_ssl_error: bool = True
    ) -> Optional[requests.Response]:
        """
        Internal method to perform HTTP request with comprehensive error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            headers: Optional custom headers
            params: Optional query parameters
            data: Optional form data
            json_data: Optional JSON data
            timeout: Optional custom timeout
            verify_ssl: Whether to verify SSL certificates
            allow_redirects: Whether to follow redirects
            retry_on_ssl_error: Whether to retry with SSL verification disabled
            
        Returns:
            Response object or None if request failed
        """
        # Extract domain for session management and rate limiting
        domain = self._extract_domain(url)
        
        # Apply rate limiting
        self._apply_rate_limit(domain)
        
        # Get session for this domain
        session = self._get_session(domain)
        
        # Prepare headers with random user agent
        request_headers = {
            'User-Agent': self._get_random_user_agent()
        }
        if headers:
            request_headers.update(headers)
        
        # Use default timeout if not specified
        request_timeout = timeout if timeout is not None else self.timeout
        
        # Update statistics
        self._update_stats('total_requests')
        
        # Retry loop with exponential backoff
        retry_count = 0
        last_error = None
        ssl_error_occurred = False
        
        while retry_count <= self.max_retries:
            try:
                # Log request attempt
                if retry_count > 0:
                    logging.debug(
                        f"[HTTPClientPool] Retry {retry_count}/{self.max_retries} for {method} {url}"
                    )
                
                # Perform request
                start_time = time.time()
                
                response = session.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    params=params,
                    data=data,
                    json=json_data,
                    timeout=request_timeout,
                    verify=verify_ssl,
                    allow_redirects=allow_redirects
                )
                
                elapsed = time.time() - start_time
                
                # Check response status
                if response.status_code >= 400:
                    logging.warning(
                        f"[HTTPClientPool] HTTP {response.status_code} for {method} {url} "
                        f"(elapsed: {elapsed:.2f}s)"
                    )
                    
                    # Handle rate limiting (429)
                    if response.status_code == 429:
                        # For LLM APIs (OpenRouter, Groq), return immediately to allow fallback
                        if 'openrouter.ai' in url or 'groq.com' in url:
                            logging.warning(
                                f"[HTTPClientPool] Rate limited (429) on LLM API - "
                                f"returning immediately for fallback"
                            )
                            self._update_stats('failed_requests')
                            return response  # Return response so LLM service can handle fallback
                        
                        # For Twitter API, return immediately to skip and move to other sources
                        if 'twitter.com' in url or 'api.twitter.com' in url:
                            logging.warning(
                                f"[HTTPClientPool] Rate limited (429) on Twitter API - "
                                f"skipping to other sources"
                            )
                            self._update_stats('failed_requests')
                            return response  # Return response so Twitter scraper can handle it
                        
                        # For other APIs, use exponential backoff
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            # Exponential backoff for rate limits (60s, 120s, 240s)
                            wait_time = 60 * (2 ** (retry_count - 1))
                            logging.warning(
                                f"[HTTPClientPool] Rate limited (429), waiting {wait_time}s before retry"
                            )
                            time.sleep(wait_time)
                            continue
                    
                    # Handle server errors (500, 502, 503, 504)
                    elif response.status_code in [500, 502, 503, 504]:
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            # Wait 30 seconds for server errors
                            logging.warning(
                                f"[HTTPClientPool] Server error ({response.status_code}), "
                                f"waiting 30s before retry"
                            )
                            time.sleep(30)
                            continue
                    
                    # Handle 404 - skip and don't retry
                    elif response.status_code == 404:
                        logging.warning(f"[HTTPClientPool] 404 Not Found: {url}")
                        self._update_stats('failed_requests')
                        return None
                
                # Success
                logging.debug(
                    f"[HTTPClientPool] {method} {url} completed "
                    f"(status: {response.status_code}, elapsed: {elapsed:.2f}s)"
                )
                
                self._update_stats('successful_requests')
                if retry_count > 0:
                    self._update_stats('retried_requests')
                
                return response
                
            except requests.exceptions.SSLError as e:
                ssl_error_occurred = True
                last_error = e
                self._update_stats('ssl_errors')
                
                # Detailed SSL error logging
                error_type = type(e).__name__
                error_msg = str(e)
                logging.warning(
                    f"[HTTPClientPool] SSL error for {url}\n"
                    f"  Error Type: {error_type}\n"
                    f"  Error Message: {error_msg}\n"
                    f"  Verify SSL: {verify_ssl}\n"
                    f"  Retry Count: {retry_count}/{self.max_retries}"
                )
                
                # Retry with SSL verification disabled if allowed
                if retry_on_ssl_error and verify_ssl:
                    logging.warning(
                        f"[HTTPClientPool] Retrying {url} with SSL verification disabled"
                    )
                    verify_ssl = False
                    retry_count += 1
                    
                    # Small delay before retry
                    time.sleep(2)
                    continue
                else:
                    # SSL error and can't retry
                    logging.error(
                        f"[HTTPClientPool] SSL error - cannot retry: {url}\n"
                        f"  Final error: {error_msg}\n"
                        f"  Suggestion: Check network connection, firewall, or system time"
                    )
                    break
            
            except requests.exceptions.Timeout as e:
                last_error = e
                retry_count += 1
                self._update_stats('timeout_errors')
                
                logging.warning(
                    f"[HTTPClientPool] Timeout for {url} "
                    f"(attempt {retry_count}/{self.max_retries})"
                )
                
                if retry_count <= self.max_retries:
                    # Exponential backoff (2s, 4s, 8s)
                    wait_time = 2 ** retry_count
                    logging.info(f"[HTTPClientPool] Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    break
            
            except requests.exceptions.ConnectionError as e:
                last_error = e
                retry_count += 1
                self._update_stats('connection_errors')
                
                # Check if it's an SSL-related connection error
                error_str = str(e).lower()
                is_ssl_related = any(keyword in error_str for keyword in ['ssl', 'certificate', 'handshake', 'tls'])
                
                if is_ssl_related:
                    logging.warning(
                        f"[HTTPClientPool] SSL-related connection error for {url}\n"
                        f"  Error: {e}\n"
                        f"  Attempt: {retry_count}/{self.max_retries}\n"
                        f"  This may be due to SSL certificate issues"
                    )
                else:
                    logging.warning(
                        f"[HTTPClientPool] Connection error for {url}\n"
                        f"  Error: {e}\n"
                        f"  Attempt: {retry_count}/{self.max_retries}"
                    )
                
                if retry_count <= self.max_retries:
                    # Exponential backoff (2s, 4s, 8s)
                    wait_time = 2 ** retry_count
                    logging.info(f"[HTTPClientPool] Waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    break
            
            except requests.exceptions.RequestException as e:
                last_error = e
                retry_count += 1
                
                logging.error(
                    f"[HTTPClientPool] Request error for {url} "
                    f"(attempt {retry_count}/{self.max_retries}): {e}"
                )
                
                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    time.sleep(wait_time)
                    continue
                else:
                    break
            
            except Exception as e:
                last_error = e
                logging.error(f"[HTTPClientPool] Unexpected error for {url}: {e}")
                break
        
        # All retries exhausted
        self._update_stats('failed_requests')
        
        if last_error:
            logging.error(
                f"[HTTPClientPool] Request failed after {retry_count} retries: {url}"
            )
            logging.error(f"[HTTPClientPool] Last error: {last_error}")
        
        return None
    
    def close_all(self):
        """Close all sessions and cleanup resources"""
        with self.session_lock:
            logging.info(f"[HTTPClientPool] Closing all sessions (count: {len(self.sessions)})")
            
            for domain, session in self.sessions.items():
                try:
                    session.close()
                    logging.debug(f"[HTTPClientPool] Closed session for domain: {domain}")
                except Exception as e:
                    logging.warning(f"[HTTPClientPool] Error closing session for {domain}: {e}")
            
            self.sessions.clear()
            logging.info("[HTTPClientPool] All sessions closed")
    
    def get_stats(self) -> Dict:
        """
        Get request statistics
        
        Returns:
            Dictionary with request statistics
        """
        with self.stats_lock:
            stats = dict(self.stats)
            
            # Calculate success rate
            total = stats.get('total_requests', 0)
            if total > 0:
                stats['success_rate'] = stats.get('successful_requests', 0) / total
                stats['failure_rate'] = stats.get('failed_requests', 0) / total
                stats['retry_rate'] = stats.get('retried_requests', 0) / total
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
                stats['retry_rate'] = 0.0
            
            return stats
    
    def reset_stats(self):
        """Reset request statistics"""
        with self.stats_lock:
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'retried_requests': 0,
                'ssl_errors': 0,
                'timeout_errors': 0,
                'connection_errors': 0
            }
            logging.info("[HTTPClientPool] Statistics reset")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all sessions"""
        self.close_all()
        return False


# ============================================================================
# TASK 6: LLM SERVICE FOR URDU TRANSLATION AND DATA EXTRACTION
# ============================================================================

class LLMService:
    """
    LLM service for Urdu translation and data extraction using OpenRouter API.
    
    Features:
    - Selective processing based on need (Urdu detection, missing fields, low quality)
    - Urdu to English translation with caching
    - Targeted field extraction for missing data
    - Batch processing for efficiency
    - Quality enhancement for low-quality incidents
    - API error handling with retry logic
    - Cost optimization through intelligent processing decisions
    
    Requirements addressed: 5, 17, 27
    """
    
    def __init__(self, api_key: str, http_client: Optional[HTTPClientPool] = None, 
                 groq_api_key: str = '', gemini_api_key: str = '', 
                 cerebras_api_key: str = '', cerebras_api_key2: str = '',
                 huggingface_api_key: str = '', huggingface_model_id: str = 'google/gemma-2b-it',
                 chatgpt_api_key: str = '', deepseek_api_key: str = '', test_mode: str = ''):
        """
        Initialize LLM service with multi-provider fallback architecture
        
        Fallback Order (FREE providers first):
        1. OpenRouter (primary - free)
        2. Groq (fallback #1 - free)
        3. Gemini (fallback #2 - free)
        4. Cerebras (fallback #3 - free, very fast)
        5. Hugging Face (fallback #4 - free)
        6. ChatGPT (fallback #5 - paid)
        7. DeepSeek (fallback #6 - paid)
        
        Args:
            api_key: OpenRouter API key
            http_client: Optional HTTPClientPool instance for requests
            groq_api_key: Groq API key (fallback #1)
            gemini_api_key: Gemini API key (fallback #2)
            cerebras_api_key: Cerebras API key (fallback #3)
            cerebras_api_key2: Cerebras backup API key
            huggingface_api_key: Hugging Face API key (fallback #4)
            chatgpt_api_key: ChatGPT API key (fallback #5)
            deepseek_api_key: DeepSeek API key (fallback #6)
        """
        # API keys
        self.api_key = api_key
        self.groq_api_key = groq_api_key
        self.gemini_api_key = gemini_api_key
        self.cerebras_api_key = cerebras_api_key
        self.cerebras_api_key2 = cerebras_api_key2
        self.huggingface_api_key = huggingface_api_key
        self.chatgpt_api_key = chatgpt_api_key
        self.deepseek_api_key = deepseek_api_key
        self.huggingface_model_id = huggingface_model_id or 'google/gemma-2b-it'
        env_test_mode = os.getenv('LLM_TEST_MODE', '').strip().lower()
        self.test_mode = (test_mode or env_test_mode or '').lower()
        
        # API URLs
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent"
        self.cerebras_api_url = "https://api.cerebras.ai/v1/chat/completions"
        self.huggingface_api_url = f"https://api-inference.huggingface.co/models/{self.huggingface_model_id}"
        self.chatgpt_api_url = "https://api.openai.com/v1/chat/completions"
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        
        self.http_client = http_client or HTTPClientPool()
        
        # Enable LLM processing if any API key is available
        self.llm_enabled = bool(self.test_mode == 'mock' or api_key or groq_api_key or gemini_api_key or 
                                cerebras_api_key or cerebras_api_key2 or
                                huggingface_api_key or chatgpt_api_key or deepseek_api_key)
        
        # Log available services
        if self.llm_enabled:
            services = []
            if api_key:
                services.append("OpenRouter")
            if groq_api_key:
                services.append("Groq")
            if gemini_api_key:
                services.append("Gemini")
            if cerebras_api_key or cerebras_api_key2:
                services.append("Cerebras")
            if huggingface_api_key:
                services.append("HuggingFace")
            if chatgpt_api_key:
                services.append("ChatGPT")
            if deepseek_api_key:
                services.append("DeepSeek")
            
            mode_suffix = " (mock mode)" if self.test_mode == 'mock' else ""
            logging.info(f"[LLMService] LLM processing enabled with {len(services)} provider(s){mode_suffix}: {', '.join(services)}")
            logging.info(f"[LLMService] Fallback order: {' → '.join(services)}")
        else:
            logging.warning("[LLMService] LLM processing disabled - no API keys configured")
        
        # Translation cache to avoid redundant API calls
        self.translation_cache: Dict[str, str] = {}
        self.cache_lock = threading.Lock()
        
        # Request tracking
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cached_responses = 0
        self.stats_lock = threading.Lock()
        
        # Processing thresholds
        self.quality_threshold = 0.5  # Process if quality below this
        self.batch_size = 5  # Number of texts to process in one batch
        
        # Fast fallback configuration - reduce retries for quicker fallback
        self.fast_fallback_retries = 1  # Only 1 retry per service before falling back
        
        # Urdu detection patterns
        self.urdu_patterns = [
            r'[\u0600-\u06FF]',  # Arabic/Urdu Unicode range
            r'[\u0750-\u077F]',  # Arabic Supplement
            r'[\uFB50-\uFDFF]',  # Arabic Presentation Forms
            r'[\uFE70-\uFEFF]'   # Arabic Presentation Forms-B
        ]
        
        logging.info(f"[LLMService] Initialized with OpenRouter API")
    
    def _mock_provider_call(self, provider_name: str, prompt: str) -> Tuple[str, bool]:
        """Generate deterministic mock responses for offline testing."""
        normalized_prompt = prompt.lower()
        if 'provide only the json' in normalized_prompt or '"area":' in normalized_prompt:
            mock_json = {
                "area": "DHA Phase 5",
                "date": datetime.now().strftime('%Y-%m-%d'),
                "time": "08:30 PM"
            }
            content = json.dumps(mock_json)
        elif 'extract and provide in json format' in normalized_prompt:
            mock_json = {
                "area": "Saddar",
                "sub_area": "Empress Market",
                "incident_date": datetime.now().strftime('%Y-%m-%d'),
                "incident_time": "09:15 PM",
                "description": "Mock quality-enhanced incident",
                "incident_type": "snatching",
                "device_model": "iPhone 15",
                "confidence": 0.95
            }
            content = json.dumps(mock_json)
        elif 'translate the following urdu' in normalized_prompt:
            content = "[Mock translation] Mobile phone was snatched in Karachi (mock output)."
        else:
            snippet = prompt.strip().splitlines()
            snippet_text = snippet[0] if snippet else ''
            snippet_text = snippet_text[:180]
            content = f"[Mock::{provider_name}] {snippet_text or 'mock response generated for empty prompt.'}"
        logging.debug(f"[LLMService] Mock response from {provider_name}")
        self._update_stats('successful_requests')
        return content, False
    
    def should_process(self, incident: Dict) -> Tuple[bool, str]:
        """
        Determine if LLM processing is needed for an incident
        
        Decision criteria:
        1. Text contains Urdu characters -> needs translation
        2. Critical fields missing (area, date, description) -> needs extraction
        3. Quality score below threshold -> needs enhancement
        4. All fields present and quality good -> skip processing
        
        Args:
            incident: Incident dictionary with text and metadata
            
        Returns:
            Tuple of (should_process: bool, reason: str)
        """
        # Skip if LLM is disabled
        if not self.llm_enabled:
            return False, "LLM processing disabled"
        
        text = incident.get('raw_text', '') or incident.get('description', '')
        
        if not text:
            return False, "No text to process"
        
        # Check for Urdu text
        if self._contains_urdu(text):
            return True, "Urdu text detected - translation needed"
        
        # Check for missing critical fields
        missing_fields = []
        if not incident.get('area') or incident.get('area') == 'Unknown':
            missing_fields.append('area')
        if not incident.get('incident_date') or incident.get('incident_date') == 'Unknown':
            missing_fields.append('date')
        if not incident.get('description') or len(incident.get('description', '')) < 20:
            missing_fields.append('description')
        
        if missing_fields:
            return True, f"Missing fields: {', '.join(missing_fields)}"
        
        # Check quality score
        quality_score = incident.get('quality_score', 0.0)
        if quality_score < self.quality_threshold:
            return True, f"Low quality score: {quality_score:.2f}"
        
        return False, "All fields present and quality acceptable"
    
    def _contains_urdu(self, text: str) -> bool:
        """
        Check if text contains Urdu/Arabic characters
        
        Args:
            text: Text to check
            
        Returns:
            True if Urdu characters detected
        """
        for pattern in self.urdu_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def translate_urdu(self, text: str) -> Optional[str]:
        """
        Translate Urdu text to English with caching
        
        Args:
            text: Urdu text to translate
            
        Returns:
            Translated English text or None if translation failed
        """
        # Skip if LLM is disabled
        if not self.llm_enabled:
            return None
        
        # Check cache first
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        with self.cache_lock:
            if cache_key in self.translation_cache:
                self._update_stats('cached_responses')
                logging.debug(f"[LLMService] Translation cache hit")
                return self.translation_cache[cache_key]
        
        # Prepare prompt for translation
        prompt = f"""Translate the following Urdu/Roman Urdu text about a crime incident in Karachi to English. 
Preserve all details including location, time, date, device models, and incident description.

Text to translate:
{text}

Provide only the English translation, no explanations."""
        
        # Call API with fallback architecture (use fast fallback for translations)
        try:
            response = self._call_api(prompt, max_tokens=500, max_retries=self.fast_fallback_retries)
            
            if response:
                translated = response.strip()
                
                # Cache the translation
                with self.cache_lock:
                    self.translation_cache[cache_key] = translated
                
                logging.info(f"[LLMService] Translated Urdu text ({len(text)} -> {len(translated)} chars)")
                return translated
            else:
                logging.warning(
                    f"[LLMService] Translation failed - all LLM services unavailable. "
                    f"Incident will be processed with original text (may be Urdu)."
                )
                return None
                
        except Exception as e:
            logging.error(f"[LLMService] Translation error: {e}")
            return None
    
    def extract_missing_fields(self, text: str, missing_fields: List[str]) -> Dict:
        """
        Use LLM to extract specific missing fields from text
        
        Args:
            text: Text to extract from
            missing_fields: List of field names to extract (e.g., ['area', 'date', 'time'])
            
        Returns:
            Dictionary with extracted fields
        """
        if not missing_fields:
            return {}
        
        # Prepare prompt for targeted extraction
        field_descriptions = {
            'area': 'the Karachi area/neighborhood where the incident occurred (e.g., DHA, Clifton, Gulshan)',
            'date': 'the date when the incident occurred in YYYY-MM-DD format',
            'time': 'the time when the incident occurred in HH:MM AM/PM format',
            'description': 'a clear description of what happened',
            'device_model': 'the phone/device model that was stolen (if mentioned)',
            'incident_type': 'the type of crime: snatching, theft, or robbery'
        }
        
        fields_to_extract = [f"- {field}: {field_descriptions.get(field, field)}" for field in missing_fields]
        
        prompt = f"""Extract the following information from this crime incident report in Karachi:

{chr(10).join(fields_to_extract)}

Text:
{text}

Provide the extracted information in JSON format. If a field is not found, use "Unknown" as the value.
Example format:
{{
  "area": "DHA Phase 5",
  "date": "2024-01-15",
  "time": "08:30 PM"
}}

Provide only the JSON, no explanations."""
        
        try:
            response = self._call_api(prompt, max_tokens=300)
            
            if response:
                # Try to parse JSON response
                try:
                    # Extract JSON from response (handle cases where LLM adds extra text)
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        extracted = json.loads(json_match.group())
                        logging.info(f"[LLMService] Extracted fields: {list(extracted.keys())}")
                        return extracted
                    else:
                        logging.warning(f"[LLMService] No JSON found in response")
                        return {}
                except json.JSONDecodeError as e:
                    logging.warning(f"[LLMService] Failed to parse JSON response: {e}")
                    return {}
            else:
                logging.warning(f"[LLMService] Field extraction failed - no response from API")
                return {}
                
        except Exception as e:
            logging.error(f"[LLMService] Field extraction error: {e}")
            return {}
    
    def enhance_quality(self, incident: Dict) -> Dict:
        """
        Use LLM to improve data quality of low-quality incidents
        
        Args:
            incident: Incident dictionary with potentially incomplete/unclear data
            
        Returns:
            Enhanced incident dictionary
        """
        text = incident.get('raw_text', '') or incident.get('description', '')
        
        if not text:
            return incident
        
        # Prepare prompt for quality enhancement
        prompt = f"""Analyze this crime incident report from Karachi and extract/enhance the following information:

Text:
{text}

Extract and provide in JSON format:
- area: Karachi neighborhood (e.g., DHA, Clifton, Gulshan, Saddar)
- sub_area: More specific location if mentioned (e.g., DHA Phase 5, Block 13)
- incident_date: Date in YYYY-MM-DD format
- incident_time: Time in HH:MM AM/PM format
- description: Clear description of what happened
- incident_type: One of: snatching, theft, robbery
- device_model: Phone/device model if mentioned
- confidence: Your confidence in the extraction (0.0 to 1.0)

If information is not available, use "Unknown". Provide only JSON, no explanations."""
        
        try:
            response = self._call_api(prompt, max_tokens=400)
            
            if response:
                # Try to parse JSON response
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        enhanced = json.loads(json_match.group())
                        
                        # Merge enhanced data with original incident
                        for key, value in enhanced.items():
                            if value and value != "Unknown" and value != "":
                                incident[key] = value
                        
                        logging.info(f"[LLMService] Enhanced incident quality")
                        return incident
                    else:
                        logging.warning(f"[LLMService] No JSON found in enhancement response")
                        return incident
                except json.JSONDecodeError as e:
                    logging.warning(f"[LLMService] Failed to parse enhancement JSON: {e}")
                    return incident
            else:
                logging.warning(f"[LLMService] Quality enhancement failed - no response from API")
                return incident
                
        except Exception as e:
            logging.error(f"[LLMService] Quality enhancement error: {e}")
            return incident
    
    def batch_process(self, texts: List[str], operation: str = 'translate') -> List[Optional[str]]:
        """
        Process multiple texts in one API call for efficiency
        
        Args:
            texts: List of texts to process
            operation: Operation type ('translate' or 'extract')
            
        Returns:
            List of processed results (same length as input)
        """
        if not texts:
            return []
        
        # For now, process individually (batch processing can be optimized later)
        # True batch processing would require more complex prompt engineering
        results = []
        
        for text in texts:
            if operation == 'translate':
                result = self.translate_urdu(text)
            else:
                result = text  # Default: return as-is
            
            results.append(result)
        
        return results
    
    def _call_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2,
        skip_openrouter: bool = False
    ) -> Optional[str]:
        """
        Call LLM API with comprehensive multi-provider fallback architecture
        
        ACTIVE Fallback Chain (FREE providers first, then PAID):
        1. OpenRouter (if available and not skipped)
        2. Groq (fallback #1)
        3. Gemini (fallback #2)
        4. Cerebras (fallback #3)
        5. Hugging Face (fallback #4)
        6. ChatGPT (paid backup)
        7. DeepSeek (paid backup)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation (0.0 = deterministic, 1.0 = creative)
            max_retries: Maximum number of retry attempts per service
            skip_openrouter: Skip OpenRouter and go directly to next fallback
            
        Returns:
            Response text or None if all services failed
        """
        if not self.llm_enabled:
            logging.warning("[LLMService] No API keys configured - skipping LLM processing")
            return None
        
        # Try OpenRouter first if available and not skipped
        if self.api_key and not skip_openrouter:
            result, should_fallback = self._call_openrouter_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.warning("[LLMService] OpenRouter failed, trying Groq fallback...")
        
        # Try Groq as fallback #1 if available
        if self.groq_api_key:
            result, _ = self._call_groq_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.warning("[LLMService] Groq failed, trying Cerebras fallback...")
        
        # Try Gemini as fallback if available (FREE)
        if self.gemini_api_key:
            result, _ = self._call_gemini_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.warning("[LLMService] Gemini failed, trying Cerebras fallback...")
        
        # Try Cerebras as fallback if available (FREE, very fast)
        if self.cerebras_api_key or self.cerebras_api_key2:
            result, _ = self._call_cerebras_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.warning("[LLMService] Cerebras failed, trying Hugging Face fallback...")
        
        # Try Hugging Face as fallback if available (FREE)
        if self.huggingface_api_key:
            result, _ = self._call_huggingface_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.warning("[LLMService] Hugging Face failed, trying ChatGPT fallback...")
        
        # Try ChatGPT as fallback #5 if available (PAID)
        if self.chatgpt_api_key:
            result, _ = self._call_chatgpt_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.warning("[LLMService] ChatGPT failed, trying DeepSeek fallback...")
        
        # Try DeepSeek as fallback #6 (last resort) if available (PAID)
        if self.deepseek_api_key:
            result, _ = self._call_deepseek_api(prompt, max_tokens, temperature, max_retries)
            if result:
                return result
            logging.error("[LLMService] All LLM providers failed - falling back to NLP-only processing")
        
        return None
    
    def _call_openrouter_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call OpenRouter API with retry logic
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts (reduced for faster fallback)
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
            - response_text: Response text or None if failed
            - should_fallback_immediately: True if rate limited (429) and should skip to Groq
        """
        
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('OpenRouter', prompt)

        # Prepare request payload
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct:free",  # Using free model (updated to 3.3)
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/karachi-crime-scraper",
            "X-Title": "Karachi Crime Scraper"
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self._update_stats('request_count')
                
                # Make API request
                response = self.http_client.post(
                    url=self.api_url,
                    json_data=payload,
                    headers=headers,
                    timeout=30,
                    verify_ssl=True,
                    retry_on_ssl_error=True
                )
                
                if response and response.status_code == 200:
                    # Parse response
                    try:
                        # Check if response has content
                        if not response.text or len(response.text.strip()) == 0:
                            logging.error(f"[LLMService] OpenRouter API returned empty response body")
                            last_error = "Empty response body"
                        else:
                            data = response.json()
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                content = data['choices'][0]['message']['content']
                                self._update_stats('successful_requests')
                                
                                logging.debug(
                                    f"[LLMService] API call successful "
                                    f"(tokens: ~{len(content.split())})"
                                )
                                
                                return content, False  # Success, no fallback needed
                            else:
                                logging.warning(f"[LLMService] Unexpected API response format: {data}")
                                last_error = "Unexpected response format"
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"[LLMService] Failed to parse API response: {e}")
                        logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                        last_error = e
                
                elif response:
                    # API returned error status
                    logging.warning(
                        f"[LLMService] OpenRouter API returned status {response.status_code}"
                    )
                    logging.debug(f"[LLMService] Response body: {response.text[:500]}")
                    logging.debug(f"[LLMService] Request headers: {headers}")
                    last_error = f"HTTP {response.status_code}"
                    
                    # Handle rate limiting - immediately fallback to Groq instead of waiting
                    if response.status_code == 429:
                        logging.warning(
                            f"[LLMService] OpenRouter rate limited (429) - "
                            f"immediately falling back to Groq instead of waiting"
                        )
                        self._update_stats('failed_requests')
                        return None, True  # Signal immediate fallback
                    
                    # Handle server errors
                    elif response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 3 * retry_count  # Linear backoff: 3s, 6s
                            logging.info(
                                f"[LLMService] Server error, waiting {wait_time}s before retry "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                else:
                    # No response from HTTP client
                    logging.error(f"[LLMService] No response from API")
                    last_error = "No response"
                
                # If we get here, request failed
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    logging.info(
                        f"[LLMService] Retrying API call in {wait_time}s "
                        f"({retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"[LLMService] API call exception: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    time.sleep(wait_time)
                else:
                    break
        
        # All retries exhausted
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] OpenRouter API call failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )
        
        return None, False  # Failed, but not a rate limit (already handled above)
    
    def _call_groq_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call Groq API with retry logic (fallback)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
            - response_text: Response text or None if failed
            - should_fallback_immediately: True if rate limited (always False for Groq as it's last fallback)
        """
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('Groq', prompt)

        # Prepare request payload for Groq
        payload = {
            "model": "llama-3.1-8b-instant",  # Fast Groq model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",  # Explicitly request JSON
            "Accept-Encoding": "identity"  # Disable compression to avoid binary responses
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self._update_stats('request_count')
                
                # Make API request
                response = self.http_client.post(
                    url=self.groq_api_url,
                    json_data=payload,
                    headers=headers,
                    timeout=30,
                    verify_ssl=True,
                    retry_on_ssl_error=True
                )
                
                if response and response.status_code == 200:
                    # Parse response
                    try:
                        # Check if response has content
                        if not response.text or len(response.text.strip()) == 0:
                            logging.error(f"[LLMService] Groq API returned empty response body")
                            last_error = "Empty response body"
                        else:
                            data = response.json()
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                content = data['choices'][0]['message']['content']
                                self._update_stats('successful_requests')
                                
                                logging.debug(
                                    f"[LLMService] Groq API call successful "
                                    f"(tokens: ~{len(content.split())})"
                                )
                                
                                return content, False  # Success
                            else:
                                logging.warning(f"[LLMService] Unexpected Groq API response format: {data}")
                                last_error = "Unexpected response format"
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"[LLMService] Failed to parse Groq API response: {e}")
                        logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                        last_error = e
                
                elif response:
                    # API returned error status
                    logging.warning(
                        f"[LLMService] Groq API returned status {response.status_code}"
                    )
                    logging.debug(f"[LLMService] Response headers: {response.headers}")
                    logging.debug(f"[LLMService] Response body (first 200 chars): {response.text[:200]}")
                    last_error = f"HTTP {response.status_code}"
                    
                    # Handle rate limiting - Groq is last fallback, so just log and fail
                    if response.status_code == 429:
                        logging.warning(
                            f"[LLMService] Groq rate limited (429) - "
                            f"no more fallbacks available, will use NLP-only processing"
                        )
                        self._update_stats('failed_requests')
                        return None, True  # Signal that we're rate limited
                    
                    # Handle server errors
                    elif response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 3 * retry_count
                            logging.info(
                                f"[LLMService] Groq server error, waiting {wait_time}s before retry "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                else:
                    # No response from HTTP client
                    logging.error(f"[LLMService] No response from Groq API")
                    last_error = "No response"
                
                # If we get here, request failed
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    logging.info(
                        f"[LLMService] Retrying Groq API call in {wait_time}s "
                        f"({retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"[LLMService] Groq API call exception: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    time.sleep(wait_time)
                else:
                    break
        
        # All retries exhausted
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] Groq API call failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )
        
        return None, False  # Failed
    
    def _call_gemini_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call Google Gemini API with retry logic (fallback #2)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
        """
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('Gemini', prompt)

        # Prepare request payload for Gemini (different format)
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        # Gemini uses API key in URL
        url_with_key = f"{self.gemini_api_url}?key={self.gemini_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self._update_stats('request_count')
                
                # Make API request
                response = self.http_client.post(
                    url=url_with_key,
                    json_data=payload,
                    headers=headers,
                    timeout=30,
                    verify_ssl=True,
                    retry_on_ssl_error=True
                )
                
                if response and response.status_code == 200:
                    try:
                        if not response.text or len(response.text.strip()) == 0:
                            logging.error(f"[LLMService] Gemini API returned empty response body")
                            last_error = "Empty response body"
                        else:
                            data = response.json()
                            
                            # Gemini response format: candidates[0].content.parts[0].text
                            if 'candidates' in data and len(data['candidates']) > 0:
                                candidate = data['candidates'][0]
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    parts = candidate['content']['parts']
                                    if len(parts) > 0 and 'text' in parts[0]:
                                        content = parts[0]['text']
                                        self._update_stats('successful_requests')
                                        
                                        logging.debug(
                                            f"[LLMService] Gemini API call successful "
                                            f"(tokens: ~{len(content.split())})"
                                        )
                                        
                                        return content, False  # Success
                            
                            logging.warning(f"[LLMService] Unexpected Gemini API response format: {data}")
                            last_error = "Unexpected response format"
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"[LLMService] Failed to parse Gemini API response: {e}")
                        logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                        last_error = e
                
                elif response:
                    logging.warning(f"[LLMService] Gemini API returned status {response.status_code}")
                    logging.debug(f"[LLMService] Response body (first 200 chars): {response.text[:200]}")
                    last_error = f"HTTP {response.status_code}"
                    
                    if response.status_code == 429:
                        logging.warning(f"[LLMService] Gemini rate limited (429)")
                        self._update_stats('failed_requests')
                        return None, True
                    
                    elif response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 3 * retry_count
                            logging.info(
                                f"[LLMService] Gemini server error, waiting {wait_time}s before retry "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                else:
                    logging.error(f"[LLMService] No response from Gemini API")
                    last_error = "No response"
                
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    logging.info(
                        f"[LLMService] Retrying Gemini API call in {wait_time}s "
                        f"({retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"[LLMService] Gemini API call exception: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    time.sleep(wait_time)
                else:
                    break
        
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] Gemini API call failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )
        
        return None, False
    
    def _call_chatgpt_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call OpenAI ChatGPT API with retry logic (fallback #3)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
        """
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('ChatGPT', prompt)

        # Prepare request payload for ChatGPT (OpenAI format)
        payload = {
            "model": "gpt-3.5-turbo",  # Using GPT-3.5 for cost efficiency
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.chatgpt_api_key}",
            "Content-Type": "application/json"
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self._update_stats('request_count')
                
                # Make API request
                response = self.http_client.post(
                    url=self.chatgpt_api_url,
                    json_data=payload,
                    headers=headers,
                    timeout=30,
                    verify_ssl=True,
                    retry_on_ssl_error=True
                )
                
                if response and response.status_code == 200:
                    try:
                        if not response.text or len(response.text.strip()) == 0:
                            logging.error(f"[LLMService] ChatGPT API returned empty response body")
                            last_error = "Empty response body"
                        else:
                            data = response.json()
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                content = data['choices'][0]['message']['content']
                                self._update_stats('successful_requests')
                                
                                logging.debug(
                                    f"[LLMService] ChatGPT API call successful "
                                    f"(tokens: ~{len(content.split())})"
                                )
                                
                                return content, False  # Success
                            else:
                                logging.warning(f"[LLMService] Unexpected ChatGPT API response format: {data}")
                                last_error = "Unexpected response format"
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"[LLMService] Failed to parse ChatGPT API response: {e}")
                        logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                        last_error = e
                
                elif response:
                    logging.warning(f"[LLMService] ChatGPT API returned status {response.status_code}")
                    logging.debug(f"[LLMService] Response body (first 200 chars): {response.text[:200]}")
                    last_error = f"HTTP {response.status_code}"
                    
                    if response.status_code == 429:
                        logging.warning(f"[LLMService] ChatGPT rate limited (429)")
                        self._update_stats('failed_requests')
                        return None, True
                    
                    elif response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 3 * retry_count
                            logging.info(
                                f"[LLMService] ChatGPT server error, waiting {wait_time}s before retry "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                else:
                    logging.error(f"[LLMService] No response from ChatGPT API")
                    last_error = "No response"
                
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    logging.info(
                        f"[LLMService] Retrying ChatGPT API call in {wait_time}s "
                        f"({retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"[LLMService] ChatGPT API call exception: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    time.sleep(wait_time)
                else:
                    break
        
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] ChatGPT API call failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )
        
        return None, False
    
    def _call_deepseek_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call DeepSeek API with retry logic (fallback #4 - last resort)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
        """
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('DeepSeek', prompt)

        # Prepare request payload for DeepSeek (OpenAI-compatible format)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self._update_stats('request_count')
                
                # Make API request
                response = self.http_client.post(
                    url=self.deepseek_api_url,
                    json_data=payload,
                    headers=headers,
                    timeout=30,
                    verify_ssl=True,
                    retry_on_ssl_error=True
                )
                
                if response and response.status_code == 200:
                    try:
                        if not response.text or len(response.text.strip()) == 0:
                            logging.error(f"[LLMService] DeepSeek API returned empty response body")
                            last_error = "Empty response body"
                        else:
                            data = response.json()
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                content = data['choices'][0]['message']['content']
                                self._update_stats('successful_requests')
                                
                                logging.debug(
                                    f"[LLMService] DeepSeek API call successful "
                                    f"(tokens: ~{len(content.split())})"
                                )
                                
                                return content, False  # Success
                            else:
                                logging.warning(f"[LLMService] Unexpected DeepSeek API response format: {data}")
                                last_error = "Unexpected response format"
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"[LLMService] Failed to parse DeepSeek API response: {e}")
                        logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                        last_error = e
                
                elif response:
                    logging.warning(f"[LLMService] DeepSeek API returned status {response.status_code}")
                    logging.debug(f"[LLMService] Response body (first 200 chars): {response.text[:200]}")
                    last_error = f"HTTP {response.status_code}"
                    
                    if response.status_code == 429:
                        logging.warning(f"[LLMService] DeepSeek rate limited (429) - no more fallbacks available")
                        self._update_stats('failed_requests')
                        return None, True
                    
                    elif response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 3 * retry_count
                            logging.info(
                                f"[LLMService] DeepSeek server error, waiting {wait_time}s before retry "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                else:
                    logging.error(f"[LLMService] No response from DeepSeek API")
                    last_error = "No response"
                
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    logging.info(
                        f"[LLMService] Retrying DeepSeek API call in {wait_time}s "
                        f"({retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"[LLMService] DeepSeek API call exception: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    time.sleep(wait_time)
                else:
                    break
        
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] DeepSeek API call failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )
        
        return None, False
    
    def _call_cerebras_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call Cerebras API with retry logic (fallback #3 - FREE, very fast)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
        """
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('Cerebras', prompt)

        # Try primary key first, then backup key
        api_keys = [self.cerebras_api_key, self.cerebras_api_key2]
        api_keys = [k for k in api_keys if k]  # Filter out empty keys
        
        if not api_keys:
            return None, False
        
        for key_index, api_key in enumerate(api_keys):
            # Prepare request payload for Cerebras (OpenAI-compatible format)
            payload = {
                "model": "llama3.1-8b",  # Fast Cerebras model
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity"  # Disable compression to avoid binary responses
            }
            
            retry_count = 0
            last_error = None
            
            while retry_count <= max_retries:
                try:
                    self._update_stats('request_count')
                    
                    # Make API request
                    response = self.http_client.post(
                        url=self.cerebras_api_url,
                        json_data=payload,
                        headers=headers,
                        timeout=30,
                        verify_ssl=True,
                        retry_on_ssl_error=True
                    )
                    
                    if response and response.status_code == 200:
                        try:
                            if not response.text or len(response.text.strip()) == 0:
                                logging.error(f"[LLMService] Cerebras API returned empty response body")
                                last_error = "Empty response body"
                            else:
                                data = response.json()
                                
                                if 'choices' in data and len(data['choices']) > 0:
                                    content = data['choices'][0]['message']['content']
                                    self._update_stats('successful_requests')
                                    
                                    logging.debug(
                                        f"[LLMService] Cerebras API call successful (key #{key_index+1}) "
                                        f"(tokens: ~{len(content.split())})"
                                    )
                                    
                                    return content, False  # Success
                                else:
                                    logging.warning(f"[LLMService] Unexpected Cerebras API response format: {data}")
                                    last_error = "Unexpected response format"
                                
                        except json.JSONDecodeError as e:
                            logging.error(f"[LLMService] Failed to parse Cerebras API response: {e}")
                            logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                            last_error = e
                    
                    elif response:
                        logging.warning(f"[LLMService] Cerebras API returned status {response.status_code}")
                        logging.debug(f"[LLMService] Response body (first 200 chars): {response.text[:200]}")
                        last_error = f"HTTP {response.status_code}"
                        
                        if response.status_code == 429:
                            logging.warning(f"[LLMService] Cerebras rate limited (429) with key #{key_index+1}")
                            self._update_stats('failed_requests')
                            # Try next key if available
                            if key_index + 1 < len(api_keys):
                                logging.info(f"[LLMService] Trying Cerebras backup key #{key_index+2}")
                                break  # Break retry loop to try next key
                            return None, True
                        
                        elif response.status_code >= 500:
                            retry_count += 1
                            if retry_count <= max_retries:
                                wait_time = 3 * retry_count
                                logging.info(
                                    f"[LLMService] Cerebras server error, waiting {wait_time}s before retry "
                                    f"({retry_count}/{max_retries})"
                                )
                                time.sleep(wait_time)
                                continue
                    else:
                        logging.error(f"[LLMService] No response from Cerebras API")
                        last_error = "No response"
                    
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        wait_time = 3 * retry_count
                        logging.info(
                            f"[LLMService] Retrying Cerebras API call in {wait_time}s "
                            f"({retry_count}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        break
                        
                except Exception as e:
                    logging.error(f"[LLMService] Cerebras API call exception: {e}")
                    last_error = e
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        wait_time = 3 * retry_count
                        time.sleep(wait_time)
                    else:
                        break
        
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] Cerebras API call failed after trying all keys. "
            f"Last error: {last_error}"
        )
        
        return None, False
    
    def _call_huggingface_api(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
        max_retries: int = 2
    ) -> Tuple[Optional[str], bool]:
        """
        Call Hugging Face Inference API with retry logic (fallback #4 - FREE)
        
        Args:
            prompt: Prompt to send to the LLM
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, should_fallback_immediately)
        """
        if self.test_mode == 'mock':
            self._update_stats('request_count')
            return self._mock_provider_call('HuggingFace', prompt)

        # Prepare request payload for Hugging Face
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                self._update_stats('request_count')
                
                # Make API request
                response = self.http_client.post(
                    url=self.huggingface_api_url,
                    json_data=payload,
                    headers=headers,
                    timeout=60,  # Hugging Face can be slower
                    verify_ssl=True,
                    retry_on_ssl_error=True
                )
                
                if response and response.status_code == 200:
                    try:
                        if not response.text or len(response.text.strip()) == 0:
                            logging.error(f"[LLMService] Hugging Face API returned empty response body")
                            last_error = "Empty response body"
                        else:
                            data = response.json()
                            
                            # Hugging Face response format: [{"generated_text": "..."}]
                            if isinstance(data, list) and len(data) > 0:
                                if 'generated_text' in data[0]:
                                    content = data[0]['generated_text']
                                    self._update_stats('successful_requests')
                                    
                                    logging.debug(
                                        f"[LLMService] Hugging Face API call successful "
                                        f"(tokens: ~{len(content.split())})"
                                    )
                                    
                                    return content, False  # Success
                            else:
                                logging.warning(f"[LLMService] Unexpected Hugging Face API response format: {data}")
                                last_error = "Unexpected response format"
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"[LLMService] Failed to parse Hugging Face API response: {e}")
                        logging.error(f"[LLMService] Response text (first 500 chars): {response.text[:500]}")
                        last_error = e
                
                elif response:
                    logging.warning(f"[LLMService] Hugging Face API returned status {response.status_code}")
                    logging.debug(f"[LLMService] Response body (first 200 chars): {response.text[:200]}")
                    last_error = f"HTTP {response.status_code}"
                    
                    if response.status_code == 429:
                        logging.warning(f"[LLMService] Hugging Face rate limited (429)")
                        self._update_stats('failed_requests')
                        return None, True
                    
                    elif response.status_code == 503:
                        # Model loading - wait longer
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 10  # Wait 10s for model to load
                            logging.info(
                                f"[LLMService] Hugging Face model loading, waiting {wait_time}s "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                    
                    elif response.status_code >= 500:
                        retry_count += 1
                        if retry_count <= max_retries:
                            wait_time = 3 * retry_count
                            logging.info(
                                f"[LLMService] Hugging Face server error, waiting {wait_time}s before retry "
                                f"({retry_count}/{max_retries})"
                            )
                            time.sleep(wait_time)
                            continue
                else:
                    logging.error(f"[LLMService] No response from Hugging Face API")
                    last_error = "No response"
                
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    logging.info(
                        f"[LLMService] Retrying Hugging Face API call in {wait_time}s "
                        f"({retry_count}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"[LLMService] Hugging Face API call exception: {e}")
                last_error = e
                retry_count += 1
                
                if retry_count <= max_retries:
                    wait_time = 3 * retry_count
                    time.sleep(wait_time)
                else:
                    break
        
        self._update_stats('failed_requests')
        logging.error(
            f"[LLMService] Hugging Face API call failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )
        
        return None, False
    
    def _update_stats(self, stat_key: str):
        """Update request statistics"""
        with self.stats_lock:
            if stat_key == 'request_count':
                self.request_count += 1
            elif stat_key == 'successful_requests':
                self.successful_requests += 1
            elif stat_key == 'failed_requests':
                self.failed_requests += 1
            elif stat_key == 'cached_responses':
                self.cached_responses += 1
    
    def get_stats(self) -> Dict:
        """
        Get LLM service statistics
        
        Returns:
            Dictionary with usage statistics
        """
        with self.stats_lock:
            stats = {
                'total_requests': self.request_count,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'cached_responses': self.cached_responses,
                'cache_size': len(self.translation_cache)
            }
            
            # Calculate success rate
            if self.request_count > 0:
                stats['success_rate'] = self.successful_requests / self.request_count
                stats['cache_hit_rate'] = self.cached_responses / (self.request_count + self.cached_responses)
            else:
                stats['success_rate'] = 0.0
                stats['cache_hit_rate'] = 0.0
            
            return stats
    
    def clear_cache(self):
        """Clear translation cache"""
        with self.cache_lock:
            cache_size = len(self.translation_cache)
            self.translation_cache.clear()
            logging.info(f"[LLMService] Cleared translation cache ({cache_size} entries)")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Log final statistics
        stats = self.get_stats()
        logging.info(f"[LLMService] Final stats: {stats}")
        return False


# ============================================================================
# TASK 2: CORE DATA MODELS AND STRUCTURES
# ============================================================================

@dataclass
class CrimeIncident:
    """
    Core data model for crime incidents with all required fields
    Includes timestamps, location, metadata, quality metrics, and geocoded coordinates
    """
    incident_id: str                    # MD5 hash of content
    source: str                         # Source name (e.g., 'Facebook', 'Twitter')
    source_url: str                     # Original URL
    scraped_at: str                     # ISO timestamp when scraped
    incident_date: str                  # YYYY-MM-DD format
    incident_time: str                  # HH:MM AM/PM format
    area: str                           # Primary area (e.g., 'DHA', 'Clifton')
    sub_area: str                       # Granular location (e.g., 'DHA Phase 5')
    location: str                       # Combined location string
    city: str                           # Always 'Karachi'
    location_for_geocoding: str         # Full address for geocoding
    latitude: Optional[float]           # Geocoded latitude
    longitude: Optional[float]          # Geocoded longitude
    description: str                    # Incident description
    incident_type: str                  # 'snatching', 'theft', 'robbery'
    device_model: str                   # Phone model if mentioned
    victim_phone: str                   # Contact if available
    imei_number: str                    # IMEI if available
    confidence_score: float             # 0.0 to 1.0
    quality_score: float                # 0.0 to 1.0
    is_statistical: bool                # Derived from aggregate stats
    is_urdu_translated: bool            # Was translated from Urdu
    duplicate_sources: List[str]        # Other sources reporting same incident
    raw_text: str                       # Original text
    post_timestamp: str                 # Platform post time
    
    def to_dict(self) -> Dict:
        """Convert incident to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrimeIncident':
        """Create incident from dictionary"""
        return cls(**data)


@dataclass
class ScrapingSource:
    """
    Data model for tracking source performance and health
    Used by the learning system to optimize scraping strategies
    """
    source_id: str                      # Unique identifier
    source_type: str                    # 'facebook', 'twitter', 'reddit', 'rss', 'news'
    url: str                            # Source URL
    active: bool                        # Whether source is currently active
    success_rate: float                 # Success rate (0.0 to 1.0)
    last_success: str                   # ISO timestamp of last successful scrape
    last_failure: str                   # ISO timestamp of last failure
    failure_count: int                  # Consecutive failure count
    avg_quality_score: float            # Average quality of incidents from this source
    total_incidents: int                # Total incidents collected
    selectors: List[str]                # CSS selectors that work for this source
    
    def to_dict(self) -> Dict:
        """Convert source to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScrapingSource':
        """Create source from dictionary"""
        return cls(**data)


@dataclass
class SourceStats:
    """Statistics for a scraping source (used within KnowledgeEntry)"""
    count: int                          # Number of incidents from this source
    quality: List[float]                # Quality scores
    success_rate: float                 # Success rate
    last_success: str                   # Last successful scrape timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SourceStats':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class KnowledgeEntry:
    """
    Data model for the learning system
    Tracks performance metrics and successful patterns across runs
    """
    runs: int                                   # Number of scraping runs
    total_incidents: int                        # Total incidents collected
    areas: Dict[str, int]                       # Area -> incident count
    sources: Dict[str, Dict]                    # Source -> SourceStats
    successful_queries: List[str]               # Queries that found incidents
    failed_queries: List[str]                   # Queries that found nothing
    selector_performance: Dict[str, float]      # Selector -> success rate
    verified_sources: List[str]                 # Credible sources for statistics
    granular_areas: List[str]                   # Comprehensive Karachi areas list
    discovered_facebook_pages: List[Dict]       # Auto-discovered Facebook pages
    discovered_facebook_groups: List[Dict]      # Auto-discovered Facebook groups
    
    def to_dict(self) -> Dict:
        """Convert knowledge entry to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeEntry':
        """Create knowledge entry from dictionary"""
        # Ensure all required fields exist with defaults
        defaults = {
            'runs': 0,
            'total_incidents': 0,
            'areas': {},
            'sources': {},
            'successful_queries': [],
            'failed_queries': [],
            'selector_performance': {},
            'verified_sources': [],
            'granular_areas': [],
            'discovered_facebook_pages': [],
            'discovered_facebook_groups': []
        }
        defaults.update(data)
        return cls(**defaults)


@dataclass
class DriverHealth:
    """
    Data model for monitoring WebDriver health and performance
    Used by DriverPoolManager to track driver status
    """
    driver_id: str                      # Unique driver identifier
    is_alive: bool                      # Whether driver is responsive
    last_check: str                     # ISO timestamp of last health check
    consecutive_failures: int           # Number of consecutive failures
    total_requests: int                 # Total requests handled
    avg_response_time: float            # Average response time in seconds
    
    def to_dict(self) -> Dict:
        """Convert driver health to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DriverHealth':
        """Create driver health from dictionary"""
        return cls(**data)


# ============================================================================
# TASK 16: KNOWLEDGE BASE AND LEARNING SYSTEM
# ============================================================================

class KnowledgeBase:
    """
    Knowledge base and learning system with persistent storage.
    
    Features:
    - Persistent JSON storage of learning data
    - Source performance tracking and ranking
    - Selector effectiveness monitoring
    - Query success rate analysis
    - Automatic area expansion when new neighborhoods discovered
    - Intelligent query generation based on historical performance
    
    Requirements: 7, 9, 21, 28
    """
    
    def __init__(self, knowledge_file: str = 'ultimate_knowledge.json'):
        """
        Initialize knowledge base with persistent storage
        
        Args:
            knowledge_file: Path to JSON file for persistent storage
        """
        self.knowledge_file = Path(knowledge_file)
        self.knowledge: KnowledgeEntry = self._load()
        
        # Initialize comprehensive Karachi areas list (100+ neighborhoods)
        self._initialize_granular_areas()
        
        # Initialize verified sources for statistical expansion
        self._initialize_verified_sources()
        
        logging.info(f"[KnowledgeBase] Initialized - Run #{self.knowledge.runs}, "
                    f"{self.knowledge.total_incidents} total incidents collected")
    
    def _load(self) -> KnowledgeEntry:
        """
        Load knowledge from persistent storage
        
        Returns:
            KnowledgeEntry with loaded or default data
        """
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    knowledge = KnowledgeEntry.from_dict(data)
                    logging.info(f"[KnowledgeBase] Loaded knowledge from {self.knowledge_file}")
                    return knowledge
            except Exception as e:
                logging.error(f"[KnowledgeBase] Failed to load knowledge file: {e}")
                logging.warning("[KnowledgeBase] Starting with fresh knowledge base")
        
        # Return default knowledge entry
        return KnowledgeEntry(
            runs=0,
            total_incidents=0,
            areas={},
            sources={},
            successful_queries=[],
            failed_queries=[],
            selector_performance={},
            verified_sources=[],
            granular_areas=[]
        )
    
    def save(self):
        """Save knowledge to persistent storage"""
        try:
            # Ensure parent directory exists
            self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty formatting
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge.to_dict(), f, indent=2, ensure_ascii=False)
            
            logging.info(f"[KnowledgeBase] Saved knowledge - Run #{self.knowledge.runs}, "
                        f"{self.knowledge.total_incidents} total incidents")
        except Exception as e:
            logging.error(f"[KnowledgeBase] Failed to save knowledge: {e}")
    
    def _initialize_granular_areas(self):
        """Initialize comprehensive Karachi areas list (100+ neighborhoods)"""
        if not self.knowledge.granular_areas:
            # Comprehensive list organized by major areas and sub-areas
            self.knowledge.granular_areas = [
                # DHA and Defence
                'DHA', 'DHA Phase 1', 'DHA Phase 2', 'DHA Phase 3', 'DHA Phase 4', 
                'DHA Phase 5', 'DHA Phase 6', 'DHA Phase 7', 'DHA Phase 8', 'Defence',
                
                # Clifton
                'Clifton', 'Clifton Block 1', 'Clifton Block 2', 'Clifton Block 3', 
                'Clifton Block 4', 'Clifton Block 5', 'Clifton Block 6', 'Clifton Block 7',
                'Clifton Block 8', 'Clifton Block 9', 'Boat Basin', 'Zamzama', 'Sea View', 'Do Darya',
                
                # Gulshan-e-Iqbal
                'Gulshan-e-Iqbal', 'Gulshan', 'Gulshan Block 1', 'Gulshan Block 2', 'Gulshan Block 3',
                'Gulshan Block 4', 'Gulshan Block 5', 'Gulshan Block 6', 'Gulshan Block 7',
                'Gulshan Block 8', 'Gulshan Block 9', 'Gulshan Block 10', 'Gulshan Block 11',
                'Gulshan Block 12', 'Gulshan Block 13', 'Gulshan Block 14', 'Gulshan Block 15',
                
                # Gulistan-e-Johar
                'Gulistan-e-Johar', 'Gulistan-e-Johar Block 1', 'Gulistan-e-Johar Block 2',
                'Gulistan-e-Johar Block 3', 'Gulistan-e-Johar Block 4', 'Gulistan-e-Johar Block 5',
                'Gulistan-e-Johar Block 6', 'Gulistan-e-Johar Block 7', 'Gulistan-e-Johar Block 8',
                'Gulistan-e-Johar Block 9', 'Gulistan-e-Johar Block 10',
                
                # Nazimabad
                'Nazimabad', 'Nazimabad Block 1', 'Nazimabad Block 2', 'Nazimabad Block 3',
                'Nazimabad Block 4', 'Nazimabad Block 5', 'Nazimabad Block 6',
                'North Nazimabad', 'Nazimabad No. 1', 'Nazimabad No. 2',
                
                # Federal B Area
                'Federal B Area', 'FB Area', 'FB Area Block 1', 'FB Area Block 2', 'FB Area Block 3',
                'FB Area Block 4', 'FB Area Block 5', 'FB Area Block 6', 'FB Area Block 7',
                'FB Area Block 8', 'FB Area Block 9', 'FB Area Block 10', 'FB Area Block 11',
                'FB Area Block 12', 'FB Area Block 13', 'FB Area Block 14', 'FB Area Block 15',
                'FB Area Block 16', 'FB Area Block 17', 'FB Area Block 18', 'FB Area Block 19',
                'FB Area Block 20',
                
                # Korangi and Landhi
                'Korangi', 'Korangi No. 1', 'Korangi No. 2', 'Korangi No. 3', 'Korangi No. 4',
                'Korangi No. 5', 'Korangi No. 6', 'Korangi Industrial Area', 'Korangi Crossing',
                'Korangi Creek',
                'Landhi', 'Landhi No. 1', 'Landhi No. 2', 'Landhi No. 3', 'Landhi No. 4',
                'Landhi No. 5', 'Landhi No. 6', 'Landhi Industrial Area',
                
                # Malir
                'Malir', 'Malir Cantt', 'Malir City', 'Malir Halt', 'Malir Colony',
                'Shah Faisal Colony', 'Jinnah Terminal', 'Airport',
                
                # North Karachi and New Karachi
                'North Karachi', 'North Karachi Sector 5', 'North Karachi Sector 7',
                'North Karachi Sector 11', 'North Karachi Buffer Zone',
                'New Karachi', 'New Karachi Sector 5', 'New Karachi Sector 11',
                
                # Saddar and Downtown
                'Saddar', 'Saddar Town', 'Empress Market', 'Regal Chowk', 'Merewether Tower',
                'Garden', 'Garden East', 'Garden West',
                
                # PECHS and Bahadurabad
                'PECHS', 'PECHS Block 2', 'PECHS Block 3', 'PECHS Block 6',
                'Bahadurabad', 'Shahrah-e-Faisal',
                
                # Liaquatabad
                'Liaquatabad', 'Liaquatabad No. 1', 'Liaquatabad No. 2', 'Liaquatabad No. 3',
                'Liaquatabad No. 4', 'Liaquatabad No. 5', 'Liaquatabad No. 6',
                'Liaquatabad No. 7', 'Liaquatabad No. 8', 'Liaquatabad No. 9',
                'Liaquatabad No. 10',
                
                # Orangi and Baldia
                'Orangi', 'Orangi Town', 'Orangi No. 1', 'Orangi No. 2', 'Orangi No. 3',
                'Orangi No. 4', 'Orangi No. 5',
                'Baldia', 'Baldia Town', 'Baldia No. 1', 'Baldia No. 2', 'Baldia No. 3',
                
                # Lyari and Keamari
                'Lyari', 'Lyari Town', 'Lyari Expressway',
                'Keamari', 'Keamari Town', 'Kemari',
                
                # SITE and Industrial Areas
                'SITE', 'SITE Area', 'SITE Industrial Area', 'SITE Super Highway',
                
                # Surjani and Scheme 33
                'Surjani Town', 'Surjani Town Sector 1', 'Surjani Town Sector 2',
                'Surjani Town Sector 3', 'Surjani Town Sector 4', 'Surjani Town Sector 5',
                'Scheme 33', 'Gulzar-e-Hijri',
                
                # Other major areas
                'Tariq Road', 'Soldier Bazaar', 'Model Colony', 'Buffer Zone', 'Sohrab Goth',
                'Gulberg', 'Gulberg Town', 'Kharadar', 'Mithadar', 'Jamshed Town', 'Jamshed Road',
                'Pak Colony', 'Manzoor Colony', 'Mehmoodabad', 'Mehmoodabad No. 1', 'Mehmoodabad No. 2',
                'Azizabad', 'Karimabad', 'Hyderi', 'Quaidabad', 'Saudabad',
                'University Road', 'Jail Road', 'Rashid Minhas Road', 'Drigh Road',
                'Korangi Road', 'Karsaz', 'Nipa', 'Nipa Chowrangi', 'Safora', 'Safora Goth',
                'Shahra-e-Pakistan', 'Mauripur', 'Hawksbay', 'Manghopir', 'Bin Qasim',
                'Bin Qasim Town', 'Gadap', 'Gadap Town', 'Saddar Cantt', 'Cantonment',
                'Faisal Cantonment'
            ]
            logging.info(f"[KnowledgeBase] Initialized {len(self.knowledge.granular_areas)} granular areas")
    
    def _initialize_verified_sources(self):
        """Initialize verified sources for statistical expansion"""
        if not self.knowledge.verified_sources:
            self.knowledge.verified_sources = [
                'dawn.com',
                'geo.tv',
                'tribune.com.pk',
                'express.com.pk',
                'thenews.com.pk',
                'karachipoliceonline',
                'sindhpolice.gov.pk',
                'twitter.com/sindhpolice',
                'facebook.com/sindhpolice'
            ]
            logging.info(f"[KnowledgeBase] Initialized {len(self.knowledge.verified_sources)} verified sources")
    
    def learn(self, incidents: List[CrimeIncident], queries: List[str], selectors_used: Optional[Dict[str, bool]] = None):
        """
        Update statistics after each scraping run
        
        Args:
            incidents: List of crime incidents collected in this run
            queries: List of queries used in this run
            selectors_used: Optional dict of selector -> success boolean for tracking effectiveness
        
        Requirements: 7, 28
        """
        # Increment run counter
        self.knowledge.runs += 1
        self.knowledge.total_incidents += len(incidents)
        
        logging.info(f"[KnowledgeBase] Learning from run #{self.knowledge.runs}: {len(incidents)} incidents")
        
        # Track area performance
        for incident in incidents:
            area = incident.area
            if area:
                if area not in self.knowledge.areas:
                    self.knowledge.areas[area] = 0
                self.knowledge.areas[area] += 1
                
                # Automatic area expansion - add new areas to granular list
                if area not in self.knowledge.granular_areas:
                    self.knowledge.granular_areas.append(area)
                    logging.info(f"[KnowledgeBase] Discovered new area: {area}")
                
                # Also add sub-area if present
                if incident.sub_area and incident.sub_area not in self.knowledge.granular_areas:
                    self.knowledge.granular_areas.append(incident.sub_area)
                    logging.info(f"[KnowledgeBase] Discovered new sub-area: {incident.sub_area}")
        
        # Track source performance
        for incident in incidents:
            source = incident.source
            if source not in self.knowledge.sources:
                self.knowledge.sources[source] = {
                    'count': 0,
                    'quality_scores': [],
                    'avg_quality': 0.0,
                    'success_rate': 1.0,
                    'last_success': datetime.now().isoformat(),
                    'failure_count': 0
                }
            
            source_stats = self.knowledge.sources[source]
            source_stats['count'] += 1
            source_stats['quality_scores'].append(incident.quality_score)
            
            # Update average quality (keep only last 100 scores for efficiency)
            if len(source_stats['quality_scores']) > 100:
                source_stats['quality_scores'] = source_stats['quality_scores'][-100:]
            
            source_stats['avg_quality'] = sum(source_stats['quality_scores']) / len(source_stats['quality_scores'])
            source_stats['last_success'] = datetime.now().isoformat()
        
        # Track query success
        for query in queries:
            # Check if query yielded any incidents
            query_incidents = [inc for inc in incidents if query.lower() in str(inc.raw_text).lower()]
            
            if query_incidents:
                # Successful query
                if query not in self.knowledge.successful_queries:
                    self.knowledge.successful_queries.append(query)
                
                # Remove from failed queries if it was there
                if query in self.knowledge.failed_queries:
                    self.knowledge.failed_queries.remove(query)
                
                logging.debug(f"[KnowledgeBase] Query success: '{query}' -> {len(query_incidents)} incidents")
            else:
                # Failed query
                if query not in self.knowledge.failed_queries and query not in self.knowledge.successful_queries:
                    self.knowledge.failed_queries.append(query)
                    logging.debug(f"[KnowledgeBase] Query failed: '{query}'")
        
        # Track selector effectiveness
        if selectors_used:
            for selector, success in selectors_used.items():
                if selector not in self.knowledge.selector_performance:
                    self.knowledge.selector_performance[selector] = 0.5  # Start with neutral score
                
                # Update selector performance using exponential moving average
                current_score = self.knowledge.selector_performance[selector]
                new_score = 1.0 if success else 0.0
                alpha = 0.3  # Learning rate
                self.knowledge.selector_performance[selector] = (alpha * new_score) + ((1 - alpha) * current_score)
        
        # Log learning summary
        logging.info(f"[KnowledgeBase] Learning summary:")
        logging.info(f"  - Total runs: {self.knowledge.runs}")
        logging.info(f"  - Total incidents: {self.knowledge.total_incidents}")
        logging.info(f"  - Tracked areas: {len(self.knowledge.areas)}")
        logging.info(f"  - Tracked sources: {len(self.knowledge.sources)}")
        logging.info(f"  - Successful queries: {len(self.knowledge.successful_queries)}")
        logging.info(f"  - Failed queries: {len(self.knowledge.failed_queries)}")
        logging.info(f"  - Granular areas: {len(self.knowledge.granular_areas)}")
    
    def get_queries(self, max_queries: int = 40) -> List[str]:
        """
        Generate intelligent search queries based on learning
        
        Prioritizes:
        1. Previously successful queries
        2. High-performing areas
        3. Diverse query patterns
        4. Granular area coverage
        
        Args:
            max_queries: Maximum number of queries to generate
        
        Returns:
            List of intelligent search queries
        
        Requirements: 7, 21, 28
        """
        queries = []
        
        # Query patterns for diversity
        patterns = [
            '{area} mobile snatching karachi',
            '{area} phone theft karachi',
            'mobile snatched {area} karachi',
            '{area} karachi mobile robbery',
            'phone stolen {area} karachi',
            '{area} mobile crime karachi today',
            '{area} street crime karachi',
            'karachi {area} phone snatching',
            '{area} karachi theft today',
            'mobile robbery {area} karachi'
        ]
        
        # 1. Start with successful queries from previous runs (top 10)
        if self.knowledge.successful_queries:
            successful_sample = self.knowledge.successful_queries[-10:]  # Most recent successful
            queries.extend(successful_sample)
            logging.debug(f"[KnowledgeBase] Added {len(successful_sample)} successful queries")
        
        # 2. Prioritize high-performing areas
        if self.knowledge.areas:
            # Sort areas by incident count (descending)
            sorted_areas = sorted(self.knowledge.areas.items(), key=lambda x: x[1], reverse=True)
            priority_areas = [area for area, count in sorted_areas[:20]]  # Top 20 areas
            logging.debug(f"[KnowledgeBase] Top areas: {priority_areas[:5]}")
        else:
            # Use default high-crime areas if no history
            priority_areas = [
                'DHA', 'Clifton', 'Gulshan', 'Nazimabad', 'Korangi', 'Malir', 'Saddar',
                'PECHS', 'Bahadurabad', 'Tariq Road', 'North Karachi', 'Orangi',
                'Liaquatabad', 'Gulistan-e-Johar', 'FB Area', 'Model Colony'
            ]
        
        # 3. Generate queries for priority areas with diverse patterns
        for area in priority_areas:
            if len(queries) >= max_queries:
                break
            
            # Use different patterns for each area
            for pattern in patterns[:3]:  # Use top 3 patterns per area
                query = pattern.format(area=area)
                
                # Avoid duplicates
                if query not in queries:
                    queries.append(query)
                
                if len(queries) >= max_queries:
                    break
        
        # 4. Add queries for newly discovered granular areas (exploration)
        if len(queries) < max_queries:
            # Find areas in granular list that haven't been queried much
            unexplored_areas = [
                area for area in self.knowledge.granular_areas
                if area not in self.knowledge.areas or self.knowledge.areas[area] < 3
            ]
            
            # Sample some unexplored areas
            import random
            sample_size = min(5, len(unexplored_areas))
            if sample_size > 0:
                sampled_areas = random.sample(unexplored_areas, sample_size)
                
                for area in sampled_areas:
                    if len(queries) >= max_queries:
                        break
                    
                    query = patterns[0].format(area=area)  # Use primary pattern
                    if query not in queries:
                        queries.append(query)
                        logging.debug(f"[KnowledgeBase] Added exploration query for: {area}")
        
        # Trim to max_queries
        queries = queries[:max_queries]
        
        logging.info(f"[KnowledgeBase] Generated {len(queries)} intelligent queries")
        logging.info(f"  - From successful history: {min(10, len(self.knowledge.successful_queries))}")
        logging.info(f"  - From high-performing areas: {len([q for q in queries if any(a in q for a in priority_areas[:5])])}")
        
        return queries
    
    def get_source_rankings(self) -> List[Tuple[str, Dict]]:
        """
        Get sources ranked by performance
        
        Returns:
            List of (source_name, stats) tuples sorted by performance
        
        Requirements: 7, 28
        """
        if not self.knowledge.sources:
            return []
        
        # Calculate performance score for each source
        source_scores = []
        for source, stats in self.knowledge.sources.items():
            # Performance score = (count * avg_quality) / (failure_count + 1)
            count = stats.get('count', 0)
            avg_quality = stats.get('avg_quality', 0.0)
            failure_count = stats.get('failure_count', 0)
            
            performance_score = (count * avg_quality) / (failure_count + 1)
            source_scores.append((source, stats, performance_score))
        
        # Sort by performance score (descending)
        source_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Return source name and stats (without score)
        return [(source, stats) for source, stats, _ in source_scores]
    
    def get_selector_rankings(self) -> List[Tuple[str, float]]:
        """
        Get CSS selectors ranked by effectiveness
        
        Returns:
            List of (selector, effectiveness_score) tuples sorted by score
        
        Requirements: 7, 28
        """
        if not self.knowledge.selector_performance:
            return []
        
        # Sort selectors by performance score (descending)
        ranked = sorted(
            self.knowledge.selector_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
    
    def should_reduce_source_frequency(self, source: str) -> bool:
        """
        Determine if a source should be scraped less frequently
        
        Args:
            source: Source name to check
        
        Returns:
            True if source should be reduced, False otherwise
        
        Requirements: 7, 28
        """
        if source not in self.knowledge.sources:
            return False
        
        stats = self.knowledge.sources[source]
        avg_quality = stats.get('avg_quality', 0.0)
        failure_count = stats.get('failure_count', 0)
        
        # Reduce frequency if:
        # 1. Average quality is below 0.4
        # 2. Failure count is high (> 5)
        if avg_quality < 0.4 or failure_count > 5:
            logging.info(f"[KnowledgeBase] Recommending reduced frequency for {source} "
                        f"(quality: {avg_quality:.2f}, failures: {failure_count})")
            return True
        
        return False
    
    def add_discovered_area(self, area: str):
        """
        Add a newly discovered area to the granular areas list
        
        Args:
            area: Area name to add
        
        Requirements: 9, 21
        """
        if area and area not in self.knowledge.granular_areas:
            self.knowledge.granular_areas.append(area)
            logging.info(f"[KnowledgeBase] Added discovered area: {area}")
    
    def discover_facebook_source(self, url: str, source_type: str, incidents_found: int, quality_avg: float):
        """
        Discover and save a new Facebook page or group that produces crime reports
        
        Args:
            url: Facebook page/group URL
            source_type: 'page' or 'group'
            incidents_found: Number of incidents found from this source
            quality_avg: Average quality score of incidents
        """
        # Determine which list to update
        source_list = (self.knowledge.discovered_facebook_pages if source_type == 'page' 
                      else self.knowledge.discovered_facebook_groups)
        
        # Check if already discovered
        existing = next((s for s in source_list if s['url'] == url), None)
        
        if existing:
            # Update existing source
            existing['incidents_found'] += incidents_found
            existing['total_scrapes'] += 1
            existing['last_scraped'] = datetime.now().isoformat()
            existing['quality_avg'] = (existing['quality_avg'] + quality_avg) / 2
            existing['success_rate'] = existing['incidents_found'] / existing['total_scrapes']
            logging.info(f"[KnowledgeBase] Updated {source_type}: {url} (total: {existing['incidents_found']} incidents)")
        else:
            # Add new source
            new_source = {
                'url': url,
                'type': source_type,
                'discovered_at': datetime.now().isoformat(),
                'last_scraped': datetime.now().isoformat(),
                'incidents_found': incidents_found,
                'total_scrapes': 1,
                'quality_avg': quality_avg,
                'success_rate': 1.0 if incidents_found > 0 else 0.0,
                'active': True
            }
            source_list.append(new_source)
            logging.info(f"[KnowledgeBase] Discovered new {source_type}: {url} ({incidents_found} incidents)")
        
        # Save immediately
        self.save()
    
    def get_discovered_facebook_sources(self, min_quality: float = 0.5, min_incidents: int = 1) -> List[Dict]:
        """
        Get all discovered Facebook sources that meet quality criteria
        
        Args:
            min_quality: Minimum average quality score
            min_incidents: Minimum number of incidents found
            
        Returns:
            List of Facebook source dictionaries
        """
        all_sources = (self.knowledge.discovered_facebook_pages + 
                      self.knowledge.discovered_facebook_groups)
        
        # Filter by quality and productivity
        filtered = [
            s for s in all_sources
            if s.get('active', True) 
            and s.get('quality_avg', 0) >= min_quality
            and s.get('incidents_found', 0) >= min_incidents
        ]
        
        # Sort by success rate and incidents found
        filtered.sort(key=lambda x: (x.get('success_rate', 0), x.get('incidents_found', 0)), reverse=True)
        
        return filtered
    
    def mark_facebook_source_inactive(self, url: str):
        """
        Mark a Facebook source as inactive (e.g., page deleted, access denied)
        
        Args:
            url: Facebook page/group URL
        """
        all_sources = (self.knowledge.discovered_facebook_pages + 
                      self.knowledge.discovered_facebook_groups)
        
        for source in all_sources:
            if source['url'] == url:
                source['active'] = False
                source['deactivated_at'] = datetime.now().isoformat()
                logging.info(f"[KnowledgeBase] Marked source as inactive: {url}")
                self.save()
                break
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive knowledge base statistics
        
        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            'runs': self.knowledge.runs,
            'total_incidents': self.knowledge.total_incidents,
            'tracked_areas': len(self.knowledge.areas),
            'tracked_sources': len(self.knowledge.sources),
            'successful_queries': len(self.knowledge.successful_queries),
            'failed_queries': len(self.knowledge.failed_queries),
            'tracked_selectors': len(self.knowledge.selector_performance),
            'verified_sources': len(self.knowledge.verified_sources),
            'granular_areas': len(self.knowledge.granular_areas),
            'discovered_facebook_pages': len(self.knowledge.discovered_facebook_pages),
            'discovered_facebook_groups': len(self.knowledge.discovered_facebook_groups),
            'top_areas': sorted(self.knowledge.areas.items(), key=lambda x: x[1], reverse=True)[:10] if self.knowledge.areas else [],
            'top_sources': [(s, st.get('count', 0)) for s, st in list(self.knowledge.sources.items())[:10]]
        }


# Backward compatibility alias
AILearning = KnowledgeBase


# ============================================================================
# TASK 17: PROGRESS TRACKER FOR RECOVERY AND IDEMPOTENCY
# ============================================================================

class ProgressTracker:
    """
    Progress tracker with checkpoint system for recovery and idempotency.
    
    Features:
    - Checkpoint system for crash recovery
    - Query completion tracking
    - Scraped ID persistence across runs
    - Run statistics tracking
    - Atomic file operations for data safety
    
    Requirements: 2, 22, 23
    """
    
    def __init__(self, progress_file: str = 'data/progress.json', dedup_cache_file: str = 'data/dedup_cache.json'):
        """
        Initialize progress tracker with persistent storage
        
        Args:
            progress_file: Path to JSON file for progress checkpoints
            dedup_cache_file: Path to JSON file for deduplication cache
        """
        self.progress_file = Path(progress_file)
        self.dedup_cache_file = Path(dedup_cache_file)
        
        # Ensure data directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self.dedup_cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Current progress state
        self.current_checkpoint = {
            'run_id': self._generate_run_id(),
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',  # running, completed, failed
            'completed_queries': [],
            'current_query': None,
            'current_source': None,
            'incidents_collected': 0,
            'sources_scraped': [],
            'errors_encountered': [],
            'last_checkpoint_time': datetime.now().isoformat()
        }
        
        # Deduplication cache (scraped IDs)
        self.scraped_ids: set = self._load_scraped_ids()
        
        # Run statistics
        self.run_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'incidents_found': 0,
            'duplicates_skipped': 0,
            'sources_attempted': 0,
            'sources_succeeded': 0,
            'queries_attempted': 0,
            'queries_succeeded': 0,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': 0
        }
        
        # Load previous checkpoint if exists (for recovery)
        self._load_previous_checkpoint()
        
        logging.info(f"[ProgressTracker] Initialized - Run ID: {self.current_checkpoint['run_id']}")
        logging.info(f"[ProgressTracker] Loaded {len(self.scraped_ids)} existing incident IDs for deduplication")
    
    def _generate_run_id(self) -> str:
        """
        Generate unique run ID
        
        Returns:
            Unique run identifier
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{random_suffix}"
    
    def _load_scraped_ids(self) -> set:
        """
        Load scraped IDs from deduplication cache
        
        Returns:
            Set of previously scraped incident IDs
        """
        if self.dedup_cache_file.exists():
            try:
                with open(self.dedup_cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    scraped_ids = set(data.get('scraped_ids', []))
                    logging.info(f"[ProgressTracker] Loaded {len(scraped_ids)} scraped IDs from cache")
                    return scraped_ids
            except Exception as e:
                logging.error(f"[ProgressTracker] Failed to load deduplication cache: {e}")
                return set()
        
        return set()
    
    def _save_scraped_ids(self):
        """Save scraped IDs to deduplication cache"""
        try:
            # Use atomic write (write to temp file, then rename)
            temp_file = self.dedup_cache_file.with_suffix('.tmp')
            
            data = {
                'scraped_ids': list(self.scraped_ids),
                'last_updated': datetime.now().isoformat(),
                'total_ids': len(self.scraped_ids)
            }
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(self.dedup_cache_file)
            
            logging.debug(f"[ProgressTracker] Saved {len(self.scraped_ids)} scraped IDs to cache")
        except Exception as e:
            logging.error(f"[ProgressTracker] Failed to save deduplication cache: {e}")
    
    def _load_previous_checkpoint(self):
        """Load previous checkpoint for recovery after crashes"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    previous_checkpoint = json.load(f)
                
                # Check if previous run was incomplete
                if previous_checkpoint.get('status') == 'running':
                    logging.warning(f"[ProgressTracker] Detected incomplete previous run: {previous_checkpoint.get('run_id')}")
                    logging.warning(f"[ProgressTracker] Previous run started at: {previous_checkpoint.get('start_time')}")
                    logging.warning(f"[ProgressTracker] Completed queries: {len(previous_checkpoint.get('completed_queries', []))}")
                    logging.warning(f"[ProgressTracker] Incidents collected: {previous_checkpoint.get('incidents_collected', 0)}")
                    
                    # Optionally restore state (for now, just log)
                    # In a full implementation, you could restore completed_queries to skip them
                    logging.info("[ProgressTracker] Starting fresh run (previous state logged for reference)")
                else:
                    logging.info(f"[ProgressTracker] Previous run completed successfully: {previous_checkpoint.get('run_id')}")
                
            except Exception as e:
                logging.error(f"[ProgressTracker] Failed to load previous checkpoint: {e}")
    
    def save_checkpoint(self, force: bool = False):
        """
        Save current progress checkpoint to persistent storage
        
        Args:
            force: Force save even if not enough time has passed
        
        Requirements: 2, 22
        """
        try:
            # Update checkpoint timestamp
            current_time = datetime.now()
            last_checkpoint = datetime.fromisoformat(self.current_checkpoint['last_checkpoint_time'])
            
            # Save checkpoint every 30 seconds or if forced
            if force or (current_time - last_checkpoint).total_seconds() >= 30:
                self.current_checkpoint['last_checkpoint_time'] = current_time.isoformat()
                
                # Use atomic write (write to temp file, then rename)
                temp_file = self.progress_file.with_suffix('.tmp')
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_checkpoint, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                temp_file.replace(self.progress_file)
                
                logging.debug(f"[ProgressTracker] Checkpoint saved - "
                            f"Queries: {len(self.current_checkpoint['completed_queries'])}, "
                            f"Incidents: {self.current_checkpoint['incidents_collected']}")
        except Exception as e:
            logging.error(f"[ProgressTracker] Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """
        Load checkpoint for recovery after crashes
        
        Returns:
            Checkpoint data if available, None otherwise
        
        Requirements: 2, 22
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                
                logging.info(f"[ProgressTracker] Loaded checkpoint from {checkpoint.get('run_id')}")
                logging.info(f"[ProgressTracker] Checkpoint status: {checkpoint.get('status')}")
                logging.info(f"[ProgressTracker] Completed queries: {len(checkpoint.get('completed_queries', []))}")
                logging.info(f"[ProgressTracker] Incidents collected: {checkpoint.get('incidents_collected', 0)}")
                
                return checkpoint
            except Exception as e:
                logging.error(f"[ProgressTracker] Failed to load checkpoint: {e}")
                return None
        
        return None
    
    def mark_query_started(self, query: str):
        """
        Mark a query as started
        
        Args:
            query: Query string that is being executed
        """
        self.current_checkpoint['current_query'] = query
        self.run_stats['queries_attempted'] += 1
        
        logging.debug(f"[ProgressTracker] Started query: {query}")
        
        # Save checkpoint
        self.save_checkpoint()
    
    def mark_query_completed(self, query: str, incidents_found: int = 0):
        """
        Mark a query as completed
        
        Args:
            query: Query string that was completed
            incidents_found: Number of incidents found by this query
        
        Requirements: 2, 22
        """
        if query not in self.current_checkpoint['completed_queries']:
            self.current_checkpoint['completed_queries'].append(query)
        
        self.current_checkpoint['current_query'] = None
        
        if incidents_found > 0:
            self.run_stats['queries_succeeded'] += 1
        
        logging.debug(f"[ProgressTracker] Completed query: {query} ({incidents_found} incidents)")
        
        # Save checkpoint
        self.save_checkpoint()
    
    def is_query_completed(self, query: str) -> bool:
        """
        Check if a query has already been completed
        
        Args:
            query: Query string to check
        
        Returns:
            True if query was already completed, False otherwise
        
        Requirements: 2, 22
        """
        return query in self.current_checkpoint['completed_queries']
    
    def mark_source_started(self, source: str):
        """
        Mark a source as started
        
        Args:
            source: Source name that is being scraped
        """
        self.current_checkpoint['current_source'] = source
        self.run_stats['sources_attempted'] += 1
        
        logging.debug(f"[ProgressTracker] Started source: {source}")
        
        # Save checkpoint
        self.save_checkpoint()
    
    def mark_source_completed(self, source: str, success: bool = True):
        """
        Mark a source as completed
        
        Args:
            source: Source name that was completed
            success: Whether the source was scraped successfully
        """
        if source not in self.current_checkpoint['sources_scraped']:
            self.current_checkpoint['sources_scraped'].append(source)
        
        self.current_checkpoint['current_source'] = None
        
        if success:
            self.run_stats['sources_succeeded'] += 1
        
        logging.debug(f"[ProgressTracker] Completed source: {source} (success: {success})")
        
        # Save checkpoint
        self.save_checkpoint()
    
    def add_incident_id(self, incident_id: str) -> bool:
        """
        Add incident ID to scraped IDs set for deduplication
        
        Args:
            incident_id: Unique incident identifier
        
        Returns:
            True if ID was new (not a duplicate), False if already exists
        
        Requirements: 22
        """
        if incident_id in self.scraped_ids:
            self.run_stats['duplicates_skipped'] += 1
            return False
        
        self.scraped_ids.add(incident_id)
        self.current_checkpoint['incidents_collected'] += 1
        self.run_stats['incidents_found'] += 1
        
        # Periodically save scraped IDs (every 10 new IDs)
        if len(self.scraped_ids) % 10 == 0:
            self._save_scraped_ids()
        
        return True
    
    def is_duplicate(self, incident_id: str) -> bool:
        """
        Check if incident ID already exists (is a duplicate)
        
        Args:
            incident_id: Unique incident identifier
        
        Returns:
            True if incident is a duplicate, False otherwise
        
        Requirements: 22
        """
        return incident_id in self.scraped_ids
    
    def record_request(self, success: bool = True):
        """
        Record a request attempt
        
        Args:
            success: Whether the request was successful
        """
        self.run_stats['total_requests'] += 1
        
        if success:
            self.run_stats['successful_requests'] += 1
        else:
            self.run_stats['failed_requests'] += 1
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        Record an error that occurred during scraping
        
        Args:
            error_type: Type of error (e.g., 'driver_crash', 'network_error')
            error_message: Error message
            context: Optional context information
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        
        self.current_checkpoint['errors_encountered'].append(error_entry)
        
        logging.debug(f"[ProgressTracker] Recorded error: {error_type} - {error_message}")
        
        # Save checkpoint after errors
        self.save_checkpoint(force=True)
    
    def mark_run_completed(self, status: str = 'completed'):
        """
        Mark the current run as completed
        
        Args:
            status: Final status ('completed' or 'failed')
        
        Requirements: 2, 23
        """
        self.current_checkpoint['status'] = status
        self.current_checkpoint['end_time'] = datetime.now().isoformat()
        
        # Update run statistics
        self.run_stats['end_time'] = datetime.now().isoformat()
        
        start_time = datetime.fromisoformat(self.run_stats['start_time'])
        end_time = datetime.fromisoformat(self.run_stats['end_time'])
        self.run_stats['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Save final checkpoint
        self.save_checkpoint(force=True)
        
        # Save final scraped IDs
        self._save_scraped_ids()
        
        logging.info(f"[ProgressTracker] Run completed with status: {status}")
        logging.info(f"[ProgressTracker] Duration: {self.run_stats['duration_seconds']:.1f} seconds")
        logging.info(f"[ProgressTracker] Incidents collected: {self.run_stats['incidents_found']}")
        logging.info(f"[ProgressTracker] Duplicates skipped: {self.run_stats['duplicates_skipped']}")
    
    def get_run_statistics(self) -> Dict:
        """
        Get comprehensive run statistics
        
        Returns:
            Dictionary with run statistics
        
        Requirements: 23
        """
        return {
            'run_id': self.current_checkpoint['run_id'],
            'status': self.current_checkpoint['status'],
            'start_time': self.run_stats['start_time'],
            'end_time': self.run_stats['end_time'],
            'duration_seconds': self.run_stats['duration_seconds'],
            'total_requests': self.run_stats['total_requests'],
            'successful_requests': self.run_stats['successful_requests'],
            'failed_requests': self.run_stats['failed_requests'],
            'success_rate': (
                self.run_stats['successful_requests'] / self.run_stats['total_requests']
                if self.run_stats['total_requests'] > 0 else 0.0
            ),
            'incidents_found': self.run_stats['incidents_found'],
            'duplicates_skipped': self.run_stats['duplicates_skipped'],
            'sources_attempted': self.run_stats['sources_attempted'],
            'sources_succeeded': self.run_stats['sources_succeeded'],
            'queries_attempted': self.run_stats['queries_attempted'],
            'queries_succeeded': self.run_stats['queries_succeeded'],
            'completed_queries': len(self.current_checkpoint['completed_queries']),
            'errors_encountered': len(self.current_checkpoint['errors_encountered']),
            'total_scraped_ids': len(self.scraped_ids)
        }
    
    def get_progress_summary(self) -> str:
        """
        Get a human-readable progress summary
        
        Returns:
            Formatted progress summary string
        """
        stats = self.get_run_statistics()
        
        summary = f"""
Progress Summary - Run {stats['run_id']}
{'='*60}
Status: {stats['status']}
Duration: {stats['duration_seconds']:.1f}s
Incidents Found: {stats['incidents_found']}
Duplicates Skipped: {stats['duplicates_skipped']}
Queries: {stats['queries_succeeded']}/{stats['queries_attempted']} successful
Sources: {stats['sources_succeeded']}/{stats['sources_attempted']} successful
Requests: {stats['successful_requests']}/{stats['total_requests']} successful ({stats['success_rate']:.1%})
Errors: {stats['errors_encountered']}
Total Scraped IDs: {stats['total_scraped_ids']}
{'='*60}
"""
        return summary
    
    def cleanup(self):
        """Cleanup and final save"""
        # Save final state
        self.save_checkpoint(force=True)
        self._save_scraped_ids()
        
        logging.info("[ProgressTracker] Cleanup completed")


class UrduTranslator:
    """Translate Urdu text using OpenRouter LLM"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.api_key = config.OPENROUTER_API_KEY
        self.api_url = config.OPENROUTER_API_URL
        self.cache = {}
        self.enabled = config.has_openrouter_api()
    
    def translate_urdu_to_english(self, text: str) -> str:
        """Translate Urdu text to English using free LLM"""
        # Check if translation is enabled
        if not self.enabled:
            logging.debug("Urdu translation disabled (no API key)")
            return text
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Check if text contains Urdu characters
        urdu_pattern = re.compile(r'[\u0600-\u06FF]')
        if not urdu_pattern.search(text):
            return text  # Already English
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "google/gemini-flash-1.5-8b",  # Free model
                "messages": [
                    {
                        "role": "user",
                        "content": f"Translate this Urdu text to English. Only provide the translation, nothing else:\n\n{text}"
                    }
                ]
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                translation = result['choices'][0]['message']['content'].strip()
                self.cache[text_hash] = translation
                logger.debug(f"[URDU] Translated: {text[:50]}... -> {translation[:50]}...")
                return translation
            else:
                logger.debug(f"Translation API error: {response.status_code}")
                return text
        
        except Exception as e:
            logger.debug(f"Translation error: {e}")
            return text


# ============================================================================
# GEOCODING SERVICE - GOOGLE MAPS API INTEGRATION
# ============================================================================

class GeocodingService:
    """
    Geocoding service using Google Maps API to convert addresses to lat/long coordinates.
    
    Features:
    - Address to coordinates conversion
    - Caching to minimize API calls
    - Karachi-specific location handling
    - Fallback coordinates for known areas
    - Rate limiting and error handling
    """
    
    def __init__(self, api_key: str, http_client: Optional[HTTPClientPool] = None):
        """
        Initialize Geocoding service
        
        Args:
            api_key: Google Maps API key
            http_client: Optional HTTPClientPool instance
        """
        self.api_key = api_key
        self.api_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.http_client = http_client or HTTPClientPool()
        self.enabled = bool(api_key)
        
        # Geocoding cache to minimize API calls
        self.geocode_cache: Dict[str, Dict[str, float]] = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'successful_geocodes': 0,
            'failed_geocodes': 0,
            'fallback_used': 0
        }
        self.stats_lock = threading.Lock()
        
        # Fallback coordinates for major Karachi areas (approximate centers)
        self.area_fallbacks = {
            'DHA': {'latitude': 24.8138, 'longitude': 67.0680},
            'Clifton': {'latitude': 24.8138, 'longitude': 67.0296},
            'Gulshan-e-Iqbal': {'latitude': 24.9207, 'longitude': 67.0820},
            'Gulistan-e-Johar': {'latitude': 24.9207, 'longitude': 67.1320},
            'Nazimabad': {'latitude': 24.9207, 'longitude': 67.0320},
            'Federal B Area': {'latitude': 24.9207, 'longitude': 67.0620},
            'Korangi': {'latitude': 24.8238, 'longitude': 67.1320},
            'Landhi': {'latitude': 24.8238, 'longitude': 67.2020},
            'Malir': {'latitude': 24.8938, 'longitude': 67.2020},
            'North Karachi': {'latitude': 24.9907, 'longitude': 67.0620},
            'New Karachi': {'latitude': 24.9607, 'longitude': 67.0420},
            'Saddar': {'latitude': 24.8607, 'longitude': 67.0220},
            'Garden': {'latitude': 24.8707, 'longitude': 67.0120},
            'PECHS': {'latitude': 24.8707, 'longitude': 67.0620},
            'Bahadurabad': {'latitude': 24.8807, 'longitude': 67.0720},
            'Liaquatabad': {'latitude': 24.9007, 'longitude': 67.0320},
            'Orangi': {'latitude': 24.9507, 'longitude': 66.9920},
            'Baldia': {'latitude': 24.9307, 'longitude': 66.9720},
            'Lyari': {'latitude': 24.8707, 'longitude': 66.9920},
            'Keamari': {'latitude': 24.8507, 'longitude': 66.9820},
            'SITE': {'latitude': 24.9107, 'longitude': 66.9920},
            'Surjani Town': {'latitude': 25.0407, 'longitude': 67.0520},
            'Scheme 33': {'latitude': 24.9707, 'longitude': 67.1020},
            'Gulzar-e-Hijri': {'latitude': 24.9807, 'longitude': 67.1220},
            'Tariq Road': {'latitude': 24.8707, 'longitude': 67.0520},
            'Soldier Bazaar': {'latitude': 24.8607, 'longitude': 67.0420},
            'Model Colony': {'latitude': 24.8907, 'longitude': 67.0820},
            'Buffer Zone': {'latitude': 24.9407, 'longitude': 67.0720},
            'Sohrab Goth': {'latitude': 25.0107, 'longitude': 67.0820},
            'Gulberg': {'latitude': 24.9307, 'longitude': 67.0920},
            'Karachi': {'latitude': 24.8607, 'longitude': 67.0011}  # Default Karachi center
        }
        
        if self.enabled:
            logging.info("[GeocodingService] Initialized with Google Maps API")
        else:
            logging.warning("[GeocodingService] Disabled - no API key configured")
    
    def geocode_location(self, area: str, sub_area: str = '', city: str = 'Karachi', country: str = 'Pakistan') -> Dict[str, Optional[float]]:
        """
        Convert location to latitude/longitude coordinates
        
        Args:
            area: Primary area (e.g., 'DHA', 'Clifton')
            sub_area: Sub-area if available (e.g., 'Phase 5', 'Block 13')
            city: City name (default: 'Karachi')
            country: Country name (default: 'Pakistan')
            
        Returns:
            Dictionary with 'latitude' and 'longitude' keys (None if geocoding failed)
        """
        # Return None coordinates if service is disabled
        if not self.enabled:
            return {'latitude': None, 'longitude': None}
        
        # Build full address
        address_parts = []
        if sub_area and sub_area != 'Unknown':
            address_parts.append(sub_area)
        if area and area != 'Unknown':
            address_parts.append(area)
        address_parts.extend([city, country])
        
        full_address = ', '.join(address_parts)
        
        # Check cache first
        cache_key = full_address.lower().strip()
        with self.cache_lock:
            if cache_key in self.geocode_cache:
                self._update_stats('cache_hits')
                logging.debug(f"[GeocodingService] Cache hit for: {full_address}")
                return self.geocode_cache[cache_key]
        
        # Try API geocoding
        self._update_stats('total_requests')
        self._update_stats('api_calls')
        
        try:
            params = {
                'address': full_address,
                'key': self.api_key
            }
            
            response = self.http_client.get(
                url=self.api_url,
                params=params,
                timeout=10,
                verify_ssl=True
            )
            
            if response and response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'OK' and data.get('results'):
                    location = data['results'][0]['geometry']['location']
                    coordinates = {
                        'latitude': location['lat'],
                        'longitude': location['lng']
                    }
                    
                    # Cache the result
                    with self.cache_lock:
                        self.geocode_cache[cache_key] = coordinates
                    
                    self._update_stats('successful_geocodes')
                    logging.info(f"[GeocodingService] Geocoded: {full_address} -> {coordinates['latitude']:.4f}, {coordinates['longitude']:.4f}")
                    
                    return coordinates
                else:
                    logging.warning(f"[GeocodingService] Geocoding failed for '{full_address}': {data.get('status')}")
                    
            else:
                logging.warning(f"[GeocodingService] API request failed for '{full_address}'")
        
        except Exception as e:
            logging.error(f"[GeocodingService] Geocoding error for '{full_address}': {e}")
        
        # Fallback to known area coordinates
        self._update_stats('failed_geocodes')
        fallback_coords = self._get_fallback_coordinates(area)
        
        if fallback_coords:
            self._update_stats('fallback_used')
            logging.info(f"[GeocodingService] Using fallback coordinates for area: {area}")
            
            # Cache the fallback
            with self.cache_lock:
                self.geocode_cache[cache_key] = fallback_coords
            
            return fallback_coords
        
        # No coordinates available
        logging.warning(f"[GeocodingService] No coordinates available for: {full_address}")
        return {'latitude': None, 'longitude': None}
    
    def _get_fallback_coordinates(self, area: str) -> Optional[Dict[str, float]]:
        """
        Get fallback coordinates for known Karachi areas
        
        Args:
            area: Area name
            
        Returns:
            Dictionary with latitude/longitude or None
        """
        if not area or area == 'Unknown':
            return self.area_fallbacks.get('Karachi')  # Default to Karachi center
        
        # Try exact match first
        if area in self.area_fallbacks:
            return self.area_fallbacks[area]
        
        # Try partial match (e.g., "DHA Phase 5" -> "DHA")
        for known_area in self.area_fallbacks:
            if known_area.lower() in area.lower() or area.lower() in known_area.lower():
                return self.area_fallbacks[known_area]
        
        # Default to Karachi center
        return self.area_fallbacks.get('Karachi')
    
    def _update_stats(self, stat_key: str):
        """Update statistics"""
        with self.stats_lock:
            self.stats[stat_key] = self.stats.get(stat_key, 0) + 1
    
    def get_statistics(self) -> Dict:
        """Get geocoding statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def save_cache(self, filepath: str = 'data/geocode_cache.json'):
        """
        Save geocoding cache to file
        
        Args:
            filepath: Path to save cache file
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with self.cache_lock:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.geocode_cache, f, indent=2)
            
            logging.info(f"[GeocodingService] Saved cache with {len(self.geocode_cache)} entries to {filepath}")
        
        except Exception as e:
            logging.error(f"[GeocodingService] Failed to save cache: {e}")
    
    def load_cache(self, filepath: str = 'data/geocode_cache.json'):
        """
        Load geocoding cache from file
        
        Args:
            filepath: Path to cache file
        """
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                with self.cache_lock:
                    self.geocode_cache = cache_data
                
                logging.info(f"[GeocodingService] Loaded cache with {len(self.geocode_cache)} entries from {filepath}")
        
        except Exception as e:
            logging.error(f"[GeocodingService] Failed to load cache: {e}")


# ============================================================================
# TASK 7: ADVANCED NLP PROCESSOR WITH MULTI-LANGUAGE SUPPORT
# ============================================================================

class AdvancedNLP:
    """
    Advanced NLP processor with multi-language support for crime incident extraction.
    
    Features:
    - Comprehensive Karachi areas list (100+ neighborhoods)
    - Primary area and sub-area identification
    - Temporal extraction (date and time) with inference
    - Entity extraction (device model, phone number, IMEI)
    - Quality scoring based on completeness and relevance
    - Language detection (Urdu/English/Roman Urdu)
    - Main processing pipeline for incident creation
    
    Requirements addressed: 16, 17, 21, 27
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize Advanced NLP processor
        
        Args:
            llm_service: Optional LLMService for Urdu translation and enhancement
        """
        self.llm_service = llm_service
        
        # Comprehensive Karachi areas list (100+ neighborhoods)
        # Organized by major areas and their sub-areas
        self.karachi_areas = {
            # DHA and Defence
            'DHA': ['DHA Phase 1', 'DHA Phase 2', 'DHA Phase 3', 'DHA Phase 4', 'DHA Phase 5', 
                    'DHA Phase 6', 'DHA Phase 7', 'DHA Phase 8', 'Defence'],
            'Clifton': ['Clifton Block 1', 'Clifton Block 2', 'Clifton Block 3', 'Clifton Block 4',
                       'Clifton Block 5', 'Clifton Block 6', 'Clifton Block 7', 'Clifton Block 8',
                       'Clifton Block 9', 'Boat Basin', 'Zamzama', 'Sea View', 'Do Darya'],
            
            # Gulshan and Gulistan-e-Johar
            'Gulshan-e-Iqbal': ['Gulshan Block 1', 'Gulshan Block 2', 'Gulshan Block 3', 
                               'Gulshan Block 4', 'Gulshan Block 5', 'Gulshan Block 6',
                               'Gulshan Block 7', 'Gulshan Block 8', 'Gulshan Block 9',
                               'Gulshan Block 10', 'Gulshan Block 11', 'Gulshan Block 12',
                               'Gulshan Block 13', 'Gulshan Block 14', 'Gulshan Block 15'],
            'Gulistan-e-Johar': ['Gulistan-e-Johar Block 1', 'Gulistan-e-Johar Block 2',
                                'Gulistan-e-Johar Block 3', 'Gulistan-e-Johar Block 4',
                                'Gulistan-e-Johar Block 5', 'Gulistan-e-Johar Block 6',
                                'Gulistan-e-Johar Block 7', 'Gulistan-e-Johar Block 8',
                                'Gulistan-e-Johar Block 9', 'Gulistan-e-Johar Block 10'],
            
            # Nazimabad
            'Nazimabad': ['Nazimabad Block 1', 'Nazimabad Block 2', 'Nazimabad Block 3',
                         'Nazimabad Block 4', 'Nazimabad Block 5', 'Nazimabad Block 6',
                         'North Nazimabad', 'Nazimabad No. 1', 'Nazimabad No. 2'],
            
            # Federal B Area
            'Federal B Area': ['FB Area Block 1', 'FB Area Block 2', 'FB Area Block 3',
                              'FB Area Block 4', 'FB Area Block 5', 'FB Area Block 6',
                              'FB Area Block 7', 'FB Area Block 8', 'FB Area Block 9',
                              'FB Area Block 10', 'FB Area Block 11', 'FB Area Block 12',
                              'FB Area Block 13', 'FB Area Block 14', 'FB Area Block 15',
                              'FB Area Block 16', 'FB Area Block 17', 'FB Area Block 18',
                              'FB Area Block 19', 'FB Area Block 20'],
            
            # Korangi and Landhi
            'Korangi': ['Korangi No. 1', 'Korangi No. 2', 'Korangi No. 3', 'Korangi No. 4',
                       'Korangi No. 5', 'Korangi No. 6', 'Korangi Industrial Area',
                       'Korangi Crossing', 'Korangi Creek'],
            'Landhi': ['Landhi No. 1', 'Landhi No. 2', 'Landhi No. 3', 'Landhi No. 4',
                      'Landhi No. 5', 'Landhi No. 6', 'Landhi Industrial Area'],
            
            # Malir
            'Malir': ['Malir Cantt', 'Malir City', 'Malir Halt', 'Malir Colony',
                     'Shah Faisal Colony', 'Jinnah Terminal', 'Airport'],
            
            # North Karachi and New Karachi
            'North Karachi': ['North Karachi Sector 5', 'North Karachi Sector 7',
                             'North Karachi Sector 11', 'North Karachi Buffer Zone'],
            'New Karachi': ['New Karachi Sector 5', 'New Karachi Sector 11'],
            
            # Saddar and Downtown
            'Saddar': ['Saddar Town', 'Empress Market', 'Regal Chowk', 'Merewether Tower'],
            'Garden': ['Garden East', 'Garden West'],
            
            # PECHS and Bahadurabad
            'PECHS': ['PECHS Block 2', 'PECHS Block 3', 'PECHS Block 6'],
            'Bahadurabad': ['Bahadurabad', 'Shahrah-e-Faisal'],
            
            # Liaquatabad
            'Liaquatabad': ['Liaquatabad No. 1', 'Liaquatabad No. 2', 'Liaquatabad No. 3',
                           'Liaquatabad No. 4', 'Liaquatabad No. 5', 'Liaquatabad No. 6',
                           'Liaquatabad No. 7', 'Liaquatabad No. 8', 'Liaquatabad No. 9',
                           'Liaquatabad No. 10'],
            
            # Orangi and Baldia
            'Orangi': ['Orangi Town', 'Orangi No. 1', 'Orangi No. 2', 'Orangi No. 3',
                      'Orangi No. 4', 'Orangi No. 5'],
            'Baldia': ['Baldia Town', 'Baldia No. 1', 'Baldia No. 2', 'Baldia No. 3'],
            
            # Lyari and Keamari
            'Lyari': ['Lyari Town', 'Lyari Expressway'],
            'Keamari': ['Keamari Town', 'Kemari'],
            
            # SITE and Industrial Areas
            'SITE': ['SITE Area', 'SITE Industrial Area', 'SITE Super Highway'],
            
            # Surjani and Scheme 33
            'Surjani Town': ['Surjani Town Sector 1', 'Surjani Town Sector 2',
                            'Surjani Town Sector 3', 'Surjani Town Sector 4',
                            'Surjani Town Sector 5'],
            'Scheme 33': ['Scheme 33', 'Gulzar-e-Hijri'],
            
            # Other major areas
            'Tariq Road': ['Tariq Road'],
            'Soldier Bazaar': ['Soldier Bazaar'],
            'Model Colony': ['Model Colony'],
            'Buffer Zone': ['Buffer Zone'],
            'Sohrab Goth': ['Sohrab Goth'],
            'Gulberg': ['Gulberg Town'],
            'Kharadar': ['Kharadar'],
            'Mithadar': ['Mithadar'],
            'Jamshed Town': ['Jamshed Town', 'Jamshed Road'],
            'Pak Colony': ['Pak Colony'],
            'Manzoor Colony': ['Manzoor Colony'],
            'Mehmoodabad': ['Mehmoodabad', 'Mehmoodabad No. 1', 'Mehmoodabad No. 2'],
            'Azizabad': ['Azizabad'],
            'Karimabad': ['Karimabad'],
            'Hyderi': ['Hyderi'],
            'Quaidabad': ['Quaidabad'],
            'Saudabad': ['Saudabad'],
            'University Road': ['University Road'],
            'Jail Road': ['Jail Road'],
            'Rashid Minhas Road': ['Rashid Minhas Road'],
            'Drigh Road': ['Drigh Road'],
            'Korangi Road': ['Korangi Road'],
            'Karsaz': ['Karsaz'],
            'Nipa': ['Nipa', 'Nipa Chowrangi'],
            'Safora': ['Safora Goth'],
            'Malir Halt': ['Malir Halt'],
            'Shahrah-e-Faisal': ['Shahrah-e-Faisal'],
            'Shahra-e-Pakistan': ['Shahra-e-Pakistan'],
            'Mauripur': ['Mauripur'],
            'Hawksbay': ['Hawksbay'],
            'Manghopir': ['Manghopir'],
            'Bin Qasim': ['Bin Qasim Town'],
            'Gadap': ['Gadap Town'],
            'Kemari': ['Kemari'],
            'Saddar Cantt': ['Saddar Cantt'],
            'Cantonment': ['Cantonment'],
            'Faisal Cantonment': ['Faisal Cantonment']
        }
        
        # Flatten areas for quick lookup
        self.all_areas = set(self.karachi_areas.keys())
        self.all_sub_areas = {}
        for parent, subs in self.karachi_areas.items():
            for sub in subs:
                self.all_sub_areas[sub.lower()] = parent
        
        # Urdu detection patterns
        self.urdu_patterns = [
            r'[\u0600-\u06FF]',  # Arabic/Urdu Unicode range
            r'[\u0750-\u077F]',  # Arabic Supplement
            r'[\uFB50-\uFDFF]',  # Arabic Presentation Forms
            r'[\uFE70-\uFEFF]'   # Arabic Presentation Forms-B
        ]
        
        # Roman Urdu keywords (distinctive Urdu words written in English script)
        self.roman_urdu_keywords = [
            'chori', 'loot', 'dacoity', 'churaya', 'gaya', 'hua', 'kar',
            'se', 'ko', 'ne', 'ka', 'ki', 'mein', 'par', 'tha', 'thi',
            'hai', 'hain', 'aur', 'ya', 'wala', 'wali', 'kiya', 'kia'
        ]
        
        # Crime keywords
        self.crime_keywords = [
            'snatch', 'snatched', 'snatching', 'theft', 'stolen', 'stole',
            'robbed', 'robbery', 'grabbed', 'loot', 'looted', 'mugging',
            'mugged', 'dacoity', 'چوری', 'لوٹ', 'چھین'
        ]
        
        # Mobile keywords
        self.mobile_keywords = [
            'mobile', 'phone', 'smartphone', 'cellphone', 'cell phone',
            'iphone', 'samsung', 'موبائل', 'فون'
        ]
        
        logging.info(
            f"[AdvancedNLP] Initialized with {len(self.all_areas)} primary areas "
            f"and {len(self.all_sub_areas)} sub-areas"
        )

    def detect_language(self, text: str) -> str:
        """
        Detect if text is Urdu, English, or Roman Urdu
        
        Args:
            text: Text to analyze
            
        Returns:
            Language identifier: 'urdu', 'english', or 'roman_urdu'
        """
        if not text:
            return 'english'
        
        # Check for Urdu Unicode characters
        for pattern in self.urdu_patterns:
            if re.search(pattern, text):
                return 'urdu'
        
        # Check for Roman Urdu (English script but Urdu words)
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return 'english'
        
        # Count Roman Urdu keywords
        roman_urdu_count = sum(1 for keyword in self.roman_urdu_keywords if keyword in text_lower)
        
        # Calculate percentage of Roman Urdu words
        # Need at least 25% of words to be Roman Urdu keywords to classify as Roman Urdu
        roman_urdu_percentage = roman_urdu_count / total_words if total_words > 0 else 0
        
        # Require both a minimum count AND percentage to avoid false positives
        if roman_urdu_count >= 5 and roman_urdu_percentage >= 0.25:
            return 'roman_urdu'
        
        # Default to English
        return 'english'
    
    def extract_area(self, text: str) -> Tuple[str, str]:
        """
        Extract primary area and sub-area from text with validation
        
        Args:
            text: Text to extract location from
            
        Returns:
            Tuple of (primary_area, sub_area)
        """
        text_lower = text.lower()
        primary_area = ''
        sub_area = ''
        
        # CRITICAL: Reject if text mentions other cities (not Karachi)
        other_cities = [
            'dadu', 'hyderabad', 'sukkur', 'larkana', 'mirpurkhas',
            'nawabshah', 'jacobabad', 'shikarpur', 'khairpur',
            'lahore', 'islamabad', 'rawalpindi', 'faisalabad', 'multan',
            'peshawar', 'quetta', 'gujranwala', 'sialkot'
        ]
        
        for city in other_cities:
            if city in text_lower:
                # Check if it's actually about that city (not just mentioned)
                city_patterns = [
                    f'{city} police',
                    f'{city} area',
                    f'in {city}',
                    f'at {city}',
                    f'from {city}',
                    f'{city},',
                ]
                if any(pattern in text_lower for pattern in city_patterns):
                    logging.debug(f"[AdvancedNLP] Rejected: Incident in {city}, not Karachi")
                    return '', ''
        
        # Verify "Karachi" is mentioned in the text
        if 'karachi' not in text_lower:
            logging.debug(f"[AdvancedNLP] Warning: 'Karachi' not mentioned in text")
            # Don't reject yet, but be more strict with area matching
        
        # First, try to find sub-areas (more specific)
        # Sort by length (longest first) to match more specific areas first
        sorted_sub_areas = sorted(self.all_sub_areas.items(), key=lambda x: len(x[0]), reverse=True)
        
        for sub_area_lower, parent_area in sorted_sub_areas:
            if sub_area_lower in text_lower:
                # Find the actual case-preserved sub-area name
                for parent, subs in self.karachi_areas.items():
                    if parent == parent_area:
                        for sub in subs:
                            if sub.lower() == sub_area_lower:
                                primary_area = parent_area
                                sub_area = sub
                                logging.debug(f"[AdvancedNLP] Found sub-area: {sub_area} in {primary_area}")
                                return primary_area, sub_area
        
        # If no sub-area found, look for primary areas
        for area in self.all_areas:
            if area.lower() in text_lower:
                primary_area = area
                logging.debug(f"[AdvancedNLP] Found primary area: {primary_area}")
                return primary_area, ''
        
        # Pattern matching for new areas not in our list (only if Karachi is mentioned)
        if 'karachi' in text_lower:
            patterns = [
                r'(?:in|at|near|from|around)\s+([A-Z][a-zA-Z\s-]+?)(?:\s+area|\s+karachi|,|\s+sector)',
                r'([A-Z][a-zA-Z\s-]+?)\s+(?:police|area|sector|block|phase)',
                r'(?:area|sector|block|phase)\s+([A-Z0-9][a-zA-Z0-9\s-]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    potential = match.group(1).strip()
                    # Validate potential area name
                    if 3 < len(potential) < 30 and not any(char.isdigit() for char in potential[:3]):
                        # Make sure it's not a city name
                        if potential.lower() not in other_cities:
                            primary_area = potential
                            logging.debug(f"[AdvancedNLP] Discovered new area: {primary_area}")
                            return primary_area, ''
        
        return '', ''
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text by removing special characters and formatting issues
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-\'\"]+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text
    
    def extract_temporal(self, text: str, post_timestamp: Optional[str] = None) -> Tuple[str, str]:
        """
        Extract or infer date and time from text
        
        Priority:
        1. Explicit date/time in text
        2. Relative time references (today, yesterday, etc.)
        3. Post timestamp if available
        4. Current time as fallback
        
        Args:
            text: Text to extract temporal information from
            post_timestamp: Optional post publication timestamp (ISO format)
            
        Returns:
            Tuple of (incident_date: YYYY-MM-DD, incident_time: HH:MM AM/PM)
        """
        incident_date = ''
        incident_time = ''
        
        # Extract date
        # Check for relative date references
        if re.search(r'\btoday\b', text, re.IGNORECASE):
            incident_date = datetime.now().strftime('%Y-%m-%d')
            logging.debug(f"[AdvancedNLP] Found relative date: today")
        elif re.search(r'\byesterday\b', text, re.IGNORECASE):
            incident_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            logging.debug(f"[AdvancedNLP] Found relative date: yesterday")
        elif re.search(r'\blast\s+night\b', text, re.IGNORECASE):
            incident_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            incident_time = '10:00 PM'
            logging.debug(f"[AdvancedNLP] Found relative date: last night")
        elif re.search(r'\bthis\s+morning\b', text, re.IGNORECASE):
            incident_date = datetime.now().strftime('%Y-%m-%d')
            incident_time = '08:00 AM'
            logging.debug(f"[AdvancedNLP] Found relative date: this morning")
        else:
            # Try explicit date patterns
            date_patterns = [
                (r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b', '%Y-%m-%d'),  # 2024-01-15
                (r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b', '%d-%m-%Y'),  # 15-01-2024
                (r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b', '%d %B %Y'),  # 15 January 2024
                (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b', '%B %d %Y'),  # January 15, 2024
            ]
            
            for pattern, date_format in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        date_str = match.group(1)
                        parsed_date = datetime.strptime(date_str, date_format)
                        incident_date = parsed_date.strftime('%Y-%m-%d')
                        logging.debug(f"[AdvancedNLP] Extracted date: {incident_date}")
                        break
                    except ValueError:
                        continue
        
        # If no date found, use post timestamp or current date
        if not incident_date:
            if post_timestamp:
                try:
                    post_dt = datetime.fromisoformat(post_timestamp.replace('Z', '+00:00'))
                    incident_date = post_dt.strftime('%Y-%m-%d')
                    logging.debug(f"[AdvancedNLP] Using post timestamp for date: {incident_date}")
                except:
                    incident_date = datetime.now().strftime('%Y-%m-%d')
            else:
                incident_date = datetime.now().strftime('%Y-%m-%d')
                logging.debug(f"[AdvancedNLP] Using current date as fallback")
        
        # Extract time
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',  # 08:30 PM
            r'(\d{1,2}\s*(?:AM|PM|am|pm))',  # 8 PM
            r'(?:at|around|approximately)\s+(\d{1,2}:\d{2})',  # at 20:30
            r'(\d{1,2}:\d{2})\s*(?:hours|hrs)',  # 20:30 hours
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                incident_time = match.group(1).upper()
                logging.debug(f"[AdvancedNLP] Extracted time: {incident_time}")
                break
        
        # If no explicit time, check for time-of-day references
        if not incident_time:
            time_of_day_map = {
                r'\bearly\s+morning\b': '06:00 AM',
                r'\bmorning\b': '08:00 AM',
                r'\bnoon\b': '12:00 PM',
                r'\bafternoon\b': '03:00 PM',
                r'\bevening\b': '06:00 PM',
                r'\bnight\b': '10:00 PM',
                r'\bmidnight\b': '12:00 AM',
                r'\bdawn\b': '05:00 AM',
                r'\bdusk\b': '07:00 PM'
            }
            
            for pattern, default_time in time_of_day_map.items():
                if re.search(pattern, text, re.IGNORECASE):
                    incident_time = default_time
                    logging.debug(f"[AdvancedNLP] Inferred time from context: {incident_time}")
                    break
        
        # If still no time, use post timestamp or mark as unknown
        if not incident_time:
            if post_timestamp:
                try:
                    post_dt = datetime.fromisoformat(post_timestamp.replace('Z', '+00:00'))
                    incident_time = post_dt.strftime('%I:%M %p')
                    logging.debug(f"[AdvancedNLP] Using post timestamp for time: {incident_time}")
                except:
                    incident_time = 'Unknown'
            else:
                incident_time = 'Unknown'
        
        return incident_date, incident_time
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract entities from text: device model, phone number, IMEI
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {
            'device_model': '',
            'victim_phone': '',
            'imei_number': ''
        }
        
        text_lower = text.lower()
        
        # CRITICAL: Don't extract device if text is about cameras/CCTV
        camera_keywords = ['camera', 'cctv', 'surveillance', 'footage', 'recording', 'street camera']
        if any(keyword in text_lower for keyword in camera_keywords):
            # Check if it's actually about mobile phones
            mobile_keywords = ['mobile', 'phone', 'cell phone', 'smartphone']
            if not any(keyword in text_lower for keyword in mobile_keywords):
                logging.debug(f"[AdvancedNLP] Skipping device extraction: Text is about cameras, not phones")
                return entities
        
        # Extract device model - only if context is about mobile phones
        device_patterns = [
            # iPhone models
            r'(iPhone\s+\d+\s*(?:Pro\s*Max|Pro|Plus|Mini)?)',
            # Samsung models
            r'(Samsung\s+Galaxy\s+[A-Z]\d+\s*(?:Plus|Ultra)?)',
            r'(Samsung\s+[A-Z]\d+)',
            # Oppo models
            r'(Oppo\s+[A-Z]\d+\s*(?:Pro)?)',
            r'(Oppo\s+Reno\s*\d*)',
            r'(Oppo\s+F\d+)',
            # Vivo models
            r'(Vivo\s+[VY]\d+\s*(?:Pro)?)',
            r'(Vivo\s+[A-Z]\d+)',
            # Xiaomi models
            r'(Xiaomi\s+(?:Redmi|Mi|Poco)\s+[A-Z0-9]+\s*(?:Pro)?)',
            r'(Redmi\s+(?:Note\s+)?\d+\s*(?:Pro)?)',
            # OnePlus models
            r'(OnePlus\s+\d+\s*(?:Pro|T)?)',
            r'(OnePlus\s+Nord\s*\d*)',
            # Huawei models
            r'(Huawei\s+[A-Z]\d+\s*(?:Pro)?)',
            r'(Huawei\s+P\d+\s*(?:Pro)?)',
            # Realme models
            r'(Realme\s+\d+\s*(?:Pro)?)',
            r'(Realme\s+[A-Z]\d+)',
            # Infinix models
            r'(Infinix\s+(?:Hot|Note|Zero)\s*\d*)',
            # Tecno models
            r'(Tecno\s+(?:Spark|Camon|Phantom)\s*\d*)'
        ]
        
        # Only extract device if mobile/phone is mentioned nearby
        for pattern in device_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Check if mobile/phone is mentioned within 50 chars of the device
                device_pos = match.start()
                context_start = max(0, device_pos - 50)
                context_end = min(len(text), device_pos + 50)
                context = text[context_start:context_end].lower()
                
                if any(keyword in context for keyword in ['mobile', 'phone', 'cell', 'smartphone']):
                    entities['device_model'] = match.group(1)
                    logging.debug(f"[AdvancedNLP] Extracted device: {entities['device_model']}")
                    break
                else:
                    logging.debug(f"[AdvancedNLP] Skipped device: {match.group(1)} (no mobile/phone context)")
        
        # If no specific model found, check for brand names (only with mobile context)
        if not entities['device_model']:
            brands = ['iPhone', 'Samsung', 'Oppo', 'Vivo', 'Xiaomi', 'OnePlus', 
                     'Huawei', 'Realme', 'Infinix', 'Tecno', 'Nokia']
            for brand in brands:
                if brand.lower() in text_lower:
                    # Verify mobile context
                    brand_pos = text_lower.find(brand.lower())
                    context_start = max(0, brand_pos - 50)
                    context_end = min(len(text), brand_pos + 50)
                    context = text[context_start:context_end].lower()
                    
                    if any(keyword in context for keyword in ['mobile', 'phone', 'cell', 'smartphone']):
                        entities['device_model'] = brand
                        logging.debug(f"[AdvancedNLP] Extracted device brand: {brand}")
                        break
        
        # Extract phone number
        phone_patterns = [
            r'\+92\s*\d{3}\s*\d{7}',  # +92 300 1234567
            r'\+92\s*\d{10}',  # +923001234567
            r'0\d{3}\s*\d{7}',  # 0300 1234567
            r'03\d{9}',  # 03001234567
            r'\d{4}-\d{7}',  # 0300-1234567
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                entities['victim_phone'] = match.group(0)
                logging.debug(f"[AdvancedNLP] Extracted phone: {entities['victim_phone']}")
                break
        
        # Extract IMEI number (15 digits)
        imei_pattern = r'\b\d{15}\b'
        match = re.search(imei_pattern, text)
        if match:
            entities['imei_number'] = match.group(0)
            logging.debug(f"[AdvancedNLP] Extracted IMEI: {entities['imei_number']}")
        
        return entities
    
    def calculate_quality_score(self, incident: Dict) -> float:
        """
        Calculate quality score based on completeness and relevance
        
        Scoring criteria:
        - Location completeness (30%): area + sub_area
        - Temporal completeness (20%): date + time
        - Description quality (30%): length and detail
        - Crime relevance (20%): crime and mobile keywords
        
        Args:
            incident: Incident dictionary with extracted fields
            
        Returns:
            Quality score from 0.0 to 1.0
        """
        score = 0.0
        
        # Location completeness (30%)
        if incident.get('area') and incident.get('area') != 'Unknown':
            score += 0.20  # Primary area
            if incident.get('sub_area') and incident.get('sub_area') != '':
                score += 0.10  # Sub-area bonus
        
        # Temporal completeness (20%)
        if incident.get('incident_date') and incident.get('incident_date') != 'Unknown':
            score += 0.10  # Date
        if incident.get('incident_time') and incident.get('incident_time') not in ['Unknown', 'Not specified']:
            score += 0.10  # Time
        
        # Description quality (30%)
        description = incident.get('description', '')
        desc_len = len(description)
        
        if desc_len > 150:
            score += 0.30  # Detailed description
        elif desc_len > 100:
            score += 0.25  # Good description
        elif desc_len > 50:
            score += 0.20  # Adequate description
        elif desc_len > 30:
            score += 0.15  # Minimal description
        elif desc_len > 20:
            score += 0.10  # Very brief
        
        # Crime relevance (20%)
        desc_lower = description.lower()
        
        # Check for crime keywords
        crime_count = sum(1 for keyword in self.crime_keywords if keyword in desc_lower)
        mobile_count = sum(1 for keyword in self.mobile_keywords if keyword in desc_lower)
        
        if crime_count > 0 and mobile_count > 0:
            score += 0.20  # Both crime and mobile mentioned
        elif crime_count > 0 or mobile_count > 0:
            score += 0.10  # Only one category mentioned
        
        # Bonus points for additional details
        if incident.get('device_model') and incident.get('device_model') != '':
            score += 0.05  # Device model specified
        
        if incident.get('victim_phone') and incident.get('victim_phone') != '':
            score += 0.03  # Contact information available
        
        if incident.get('imei_number') and incident.get('imei_number') != '':
            score += 0.02  # IMEI available
        
        # Cap at 1.0
        final_score = min(score, 1.0)
        
        logging.debug(f"[AdvancedNLP] Quality score: {final_score:.2f}")
        
        return final_score
    
    def process_text(
        self,
        text: str,
        source: str,
        url: str,
        post_timestamp: Optional[str] = None
    ) -> Optional[CrimeIncident]:
        """
        Main processing pipeline for creating crime incidents from text
        
        Pipeline steps:
        1. Detect language
        2. Translate Urdu if needed (using LLM service)
        3. Check crime relevance
        4. Extract location (area and sub-area)
        5. Extract temporal information (date and time)
        6. Extract entities (device, phone, IMEI)
        7. Determine incident type
        8. Calculate quality score
        9. Create CrimeIncident if quality threshold met
        
        Args:
            text: Raw text to process
            source: Source name (e.g., 'Facebook', 'Twitter')
            url: Source URL
            post_timestamp: Optional post publication timestamp (ISO format)
            
        Returns:
            CrimeIncident object if valid incident found, None otherwise
        """
        try:
            if not text or len(text) < 20:
                logging.debug(f"[AdvancedNLP] Text too short: {len(text)} chars")
                return None
            
            original_text = text
            was_translated = False
            
            # Step 1: Detect language
            language = self.detect_language(text)
            logging.debug(f"[AdvancedNLP] Detected language: {language}")
            
            # Step 2: Translate Urdu if needed (with fallback to NLP-only processing)
            if language in ['urdu', 'roman_urdu'] and self.llm_service and self.llm_service.llm_enabled:
                translated = self.llm_service.translate_urdu(text)
                if translated:
                    text = translated
                    was_translated = True
                    logging.info(f"[AdvancedNLP] Translated {language} text to English")
                else:
                    logging.warning(
                        f"[AdvancedNLP] Translation failed for {language} text - "
                        f"continuing with NLP-only processing (may have reduced accuracy)"
                    )
                    # Continue processing with original text - NLP will do its best
            
            # Step 3: Filter out statistical/aggregate reports
            text_lower = text.lower()
            
            # Reject statistical reports and aggregate data
            statistical_keywords = [
                'hotspot', 'hot spot', 'hot-spot',
                'statistics', 'statistical', 'stats',
                'report shows', 'according to report', 'data shows',
                'survey', 'study shows', 'research shows',
                'total of', 'total incidents', 'total cases',
                'in total', 'altogether', 'combined',
                'multiple incidents', 'several incidents',
                'number of incidents', 'incidents reported',
                'cases reported', 'complaints received'
            ]
            
            if any(keyword in text_lower for keyword in statistical_keywords):
                logging.debug(f"[AdvancedNLP] Rejected: Statistical/aggregate report detected")
                return None
            
            # Reject if text mentions multiple numbers (likely statistics)
            number_pattern = r'\b\d{2,}\b'  # Numbers with 2+ digits
            numbers = re.findall(number_pattern, text)
            if len(numbers) >= 3:  # Multiple large numbers = likely statistics
                logging.debug(f"[AdvancedNLP] Rejected: Multiple numbers detected (likely statistics)")
                return None
            
            # Step 4: Check crime relevance
            has_crime = any(keyword in text_lower for keyword in self.crime_keywords)
            has_mobile = any(keyword in text_lower for keyword in self.mobile_keywords)
            
            if not (has_crime and has_mobile):
                logging.debug(f"[AdvancedNLP] Not crime-related (crime={has_crime}, mobile={has_mobile})")
                return None
            
            # Reject if text is about cameras/CCTV (not mobile phones)
            camera_keywords = ['camera', 'cctv', 'surveillance', 'footage', 'recording']
            if any(keyword in text_lower for keyword in camera_keywords) and 'mobile' not in text_lower and 'phone' not in text_lower:
                logging.debug(f"[AdvancedNLP] Rejected: About cameras/CCTV, not mobile phones")
                return None
            
            # Step 4: Extract location
            primary_area, sub_area = self.extract_area(text)
            
            if not primary_area:
                logging.debug(f"[AdvancedNLP] No area found in text")
                return None
            
            # Step 5: Extract temporal information
            incident_date, incident_time = self.extract_temporal(text, post_timestamp)
            
            # Step 6: Extract entities
            entities = self.extract_entities(text)
            
            # Step 7: Determine incident type
            incident_type = 'snatching'  # Default
            if 'robbery' in text_lower or 'robbed' in text_lower or 'dacoity' in text_lower:
                incident_type = 'robbery'
            elif 'theft' in text_lower or 'stolen' in text_lower or 'stole' in text_lower:
                incident_type = 'theft'
            elif 'snatch' in text_lower or 'grabbed' in text_lower:
                incident_type = 'snatching'
            
            # Build location string
            if sub_area:
                location = f"{sub_area}, {primary_area}"
                location_for_geocoding = f"{sub_area}, {primary_area}, Karachi, Pakistan"
            else:
                location = primary_area
                location_for_geocoding = f"{primary_area}, Karachi, Pakistan"
            
            # Clean the description text
            cleaned_description = self.clean_text(text[:500])
            
            # Create incident record
            incident_data = {
                'incident_id': hashlib.md5(text.encode('utf-8')).hexdigest()[:16],
                'source': source,
                'source_url': url,
                'scraped_at': datetime.now().isoformat(),
                'incident_date': incident_date,
                'incident_time': incident_time,
                'area': primary_area,
                'sub_area': sub_area,
                'location': location,
                'city': 'Karachi',
                'location_for_geocoding': location_for_geocoding,
                'latitude': None,  # Will be geocoded later
                'longitude': None,  # Will be geocoded later
                'description': cleaned_description,
                'incident_type': incident_type,
                'device_model': entities['device_model'],
                'victim_phone': entities['victim_phone'],
                'imei_number': entities['imei_number'],
                'confidence_score': 0.75,  # Base confidence
                'quality_score': 0.0,  # Will be calculated
                'is_statistical': False,  # Will be set by StatisticalExpander
                'is_urdu_translated': was_translated,
                'duplicate_sources': [],  # Will be populated by deduplication system
                'raw_text': original_text[:1000],
                'post_timestamp': post_timestamp or datetime.now().isoformat()
            }
            
            # Step 8: Calculate quality score
            incident_data['quality_score'] = self.calculate_quality_score(incident_data)
            
            # Step 9: Check quality threshold
            if incident_data['quality_score'] < 0.35:  # Increased from 0.25 to 0.35 for better quality
                logging.debug(
                    f"[AdvancedNLP] Quality score too low: {incident_data['quality_score']:.2f}"
                )
                return None
            
            # Step 10: Final validation checks
            # Reject if description is too generic or suspicious
            desc_lower = cleaned_description.lower()
            suspicious_patterns = [
                'click here', 'read more', 'full story', 'breaking news',
                'watch video', 'subscribe', 'follow us', 'share this',
                'advertisement', 'sponsored'
            ]
            if any(pattern in desc_lower for pattern in suspicious_patterns):
                logging.debug(f"[AdvancedNLP] Rejected: Suspicious content (ads/clickbait)")
                return None
            
            # Reject if description is too short after cleaning
            if len(cleaned_description) < 30:
                logging.debug(f"[AdvancedNLP] Rejected: Description too short after cleaning ({len(cleaned_description)} chars)")
                return None
            
            # Create CrimeIncident object
            incident = CrimeIncident(**incident_data)
            
            logging.info(
                f"[AdvancedNLP] Created incident: {incident.area} - "
                f"{incident.incident_type} - Quality: {incident.quality_score:.2f}"
            )
            
            return incident
            
        except Exception as e:
            logging.error(f"[AdvancedNLP] Error processing text: {e}")
            logging.debug(f"[AdvancedNLP] Text: {text[:200]}")
            return None
    
    # Legacy method for backward compatibility
    def process(self, text: str, source: str, url: str) -> Optional[CrimeIncident]:
        """Legacy method - calls process_text for backward compatibility"""
        return self.process_text(text, source, url)


# ============================================================================
# TASK 8: DEDUPLICATION SYSTEM WITH SIMILARITY SCORING
# ============================================================================

class DeduplicationSystem:
    """
    Advanced duplicate detection using multiple strategies.
    
    Features:
    - Content-based hashing (MD5 of description + location + timestamp)
    - Fuzzy text matching using Levenshtein distance (threshold: 85%)
    - Temporal proximity checking (48-hour window)
    - Cross-source duplicate detection
    - Idempotent operations (load existing IDs from output file)
    - Duplicate source merging
    
    Requirements addressed: 19, 22, 27
    """
    
    def __init__(self, similarity_threshold: float = 0.85, temporal_window_hours: int = 48):
        """
        Initialize deduplication system
        
        Args:
            similarity_threshold: Minimum similarity score to consider duplicates (default: 0.85)
            temporal_window_hours: Time window for temporal proximity checking (default: 48 hours)
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window_hours = temporal_window_hours
        
        # Set of seen incident IDs for fast lookup
        self.seen_ids: set = set()
        
        # Recent incidents for similarity comparison (within temporal window)
        self.recent_incidents: List[CrimeIncident] = []
        
        # Cache for similarity calculations to avoid redundant computations
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        logging.info(
            f"[Deduplication] Initialized with similarity_threshold={similarity_threshold}, "
            f"temporal_window={temporal_window_hours}h"
        )
    
    def generate_id(self, incident: CrimeIncident) -> str:
        """
        Generate unique ID using content-based hashing
        Uses MD5 hash of description + location + timestamp
        
        Args:
            incident: CrimeIncident object
            
        Returns:
            MD5 hash string (16 characters)
        """
        # Normalize text for consistent hashing
        description = incident.description.lower().strip()
        location = f"{incident.area}_{incident.sub_area}".lower().strip()
        timestamp = f"{incident.incident_date}_{incident.incident_time}"
        
        # Create composite string for hashing
        composite = f"{description}|{location}|{timestamp}"
        
        # Generate MD5 hash
        hash_obj = hashlib.md5(composite.encode('utf-8'))
        incident_id = hash_obj.hexdigest()[:16]
        
        logging.debug(f"[Deduplication] Generated ID: {incident_id}")
        
        return incident_id
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using Levenshtein distance or difflib fallback
        
        Uses normalized Levenshtein distance to compute similarity score:
        similarity = 1 - (levenshtein_distance / max_length)
        
        Falls back to difflib.SequenceMatcher if python-Levenshtein is not installed.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score from 0.0 (completely different) to 1.0 (identical)
        """
        # Check cache first
        cache_key = (text1, text2) if text1 < text2 else (text2, text1)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Handle empty strings
        if not t1 or not t2:
            return 0.0
        
        # Calculate similarity
        if HAS_LEVENSHTEIN:
            # Use Levenshtein distance (faster and more accurate)
            distance = Levenshtein.distance(t1, t2)
            max_length = max(len(t1), len(t2))
            
            if max_length == 0:
                similarity = 1.0
            else:
                similarity = 1.0 - (distance / max_length)
            
            logging.debug(f"[Deduplication] Similarity (Levenshtein): {similarity:.3f} (distance={distance}, max_len={max_length})")
        else:
            # Fallback to difflib (slower but works without external library)
            similarity = difflib.SequenceMatcher(None, t1, t2).ratio()
            logging.debug(f"[Deduplication] Similarity (difflib): {similarity:.3f}")
        
        # Cache the result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _is_within_temporal_window(self, incident1: CrimeIncident, incident2: CrimeIncident) -> bool:
        """
        Check if two incidents are within the temporal proximity window
        
        Args:
            incident1: First incident
            incident2: Second incident
            
        Returns:
            True if incidents are within temporal window, False otherwise
        """
        try:
            # Parse incident dates
            date1 = datetime.fromisoformat(incident1.incident_date) if 'T' in incident1.incident_date else datetime.strptime(incident1.incident_date, '%Y-%m-%d')
            date2 = datetime.fromisoformat(incident2.incident_date) if 'T' in incident2.incident_date else datetime.strptime(incident2.incident_date, '%Y-%m-%d')
            
            # Calculate time difference
            time_diff = abs((date1 - date2).total_seconds() / 3600)  # Convert to hours
            
            within_window = time_diff <= self.temporal_window_hours
            
            logging.debug(
                f"[Deduplication] Temporal check: {time_diff:.1f}h "
                f"(window={self.temporal_window_hours}h, within={within_window})"
            )
            
            return within_window
            
        except Exception as e:
            logging.warning(f"[Deduplication] Error parsing dates for temporal check: {e}")
            # If we can't parse dates, assume they're within window to be safe
            return True
    
    def is_duplicate(self, incident: CrimeIncident) -> Tuple[bool, Optional[str]]:
        """
        Check if incident is a duplicate using multiple detection strategies
        
        Detection strategies:
        1. Hash-based: Check if incident ID already exists
        2. Similarity-based: Compare description with recent incidents
        3. Temporal proximity: Only compare with incidents within 48-hour window
        
        Args:
            incident: CrimeIncident to check
            
        Returns:
            Tuple of (is_duplicate: bool, duplicate_id: Optional[str])
            - is_duplicate: True if incident is a duplicate
            - duplicate_id: ID of the original incident if duplicate found, None otherwise
        """
        # Strategy 1: Hash-based detection (fast)
        incident_id = self.generate_id(incident)
        
        if incident_id in self.seen_ids:
            logging.info(f"[Deduplication] Duplicate found (hash match): {incident_id}")
            return True, incident_id
        
        # Strategy 2: Similarity-based detection with temporal proximity
        for existing_incident in self.recent_incidents:
            # Check temporal proximity first (fast filter)
            if not self._is_within_temporal_window(incident, existing_incident):
                continue
            
            # Check location match (fast filter)
            if incident.area != existing_incident.area:
                continue
            
            # Calculate description similarity (slower)
            similarity = self.calculate_similarity(
                incident.description,
                existing_incident.description
            )
            
            # Check if similarity exceeds threshold
            if similarity >= self.similarity_threshold:
                existing_id = self.generate_id(existing_incident)
                logging.info(
                    f"[Deduplication] Duplicate found (similarity={similarity:.3f}): "
                    f"{incident.area} - {incident.incident_type}"
                )
                logging.debug(f"[Deduplication] Original ID: {existing_id}")
                return True, existing_id
        
        # Not a duplicate
        logging.debug(f"[Deduplication] Not a duplicate: {incident_id}")
        return False, None
    
    def merge_duplicates(self, original: CrimeIncident, duplicate: CrimeIncident) -> CrimeIncident:
        """
        Merge duplicate incidents by combining source URLs
        
        When a duplicate is detected, we merge the source information
        to track which outlets reported the same incident.
        
        Args:
            original: Original incident
            duplicate: Duplicate incident to merge
            
        Returns:
            Updated original incident with merged sources
        """
        # Add duplicate source to the list if not already present
        if duplicate.source_url not in original.duplicate_sources:
            original.duplicate_sources.append(duplicate.source_url)
            logging.info(
                f"[Deduplication] Merged source: {duplicate.source} -> {original.incident_id}"
            )
        
        # Update confidence score (average of both)
        original.confidence_score = (original.confidence_score + duplicate.confidence_score) / 2
        
        # Update quality score (take maximum)
        original.quality_score = max(original.quality_score, duplicate.quality_score)
        
        logging.debug(
            f"[Deduplication] Updated scores - confidence: {original.confidence_score:.2f}, "
            f"quality: {original.quality_score:.2f}"
        )
        
        return original
    
    def add_incident(self, incident: CrimeIncident) -> bool:
        """
        Add incident to the deduplication system
        
        This method should be called after checking for duplicates
        to register the incident in the system.
        
        Args:
            incident: CrimeIncident to add
            
        Returns:
            True if incident was added, False if it was a duplicate
        """
        # Check if duplicate
        is_dup, dup_id = self.is_duplicate(incident)
        
        if is_dup:
            return False
        
        # Generate and store ID
        incident_id = self.generate_id(incident)
        incident.incident_id = incident_id
        self.seen_ids.add(incident_id)
        
        # Add to recent incidents for similarity comparison
        self.recent_incidents.append(incident)
        
        # Clean up old incidents outside temporal window
        self._cleanup_old_incidents()
        
        logging.debug(
            f"[Deduplication] Added incident: {incident_id} "
            f"(recent_count={len(self.recent_incidents)})"
        )
        
        return True
    
    def _cleanup_old_incidents(self):
        """
        Remove incidents outside the temporal window from recent_incidents list
        This keeps memory usage bounded and improves performance
        """
        if not self.recent_incidents:
            return
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=self.temporal_window_hours)
        
        # Filter incidents within temporal window
        original_count = len(self.recent_incidents)
        
        self.recent_incidents = [
            inc for inc in self.recent_incidents
            if self._is_incident_recent(inc, cutoff_time)
        ]
        
        removed_count = original_count - len(self.recent_incidents)
        
        if removed_count > 0:
            logging.debug(
                f"[Deduplication] Cleaned up {removed_count} old incidents "
                f"(remaining: {len(self.recent_incidents)})"
            )
    
    def _is_incident_recent(self, incident: CrimeIncident, cutoff_time: datetime) -> bool:
        """
        Check if incident is within the temporal window
        
        Args:
            incident: Incident to check
            cutoff_time: Cutoff datetime
            
        Returns:
            True if incident is recent, False otherwise
        """
        try:
            # Parse incident date
            if 'T' in incident.incident_date:
                incident_time = datetime.fromisoformat(incident.incident_date)
            else:
                incident_time = datetime.strptime(incident.incident_date, '%Y-%m-%d')
            
            return incident_time >= cutoff_time
            
        except Exception as e:
            logging.debug(f"[Deduplication] Error parsing date for cleanup: {e}")
            # Keep incident if we can't parse date
            return True
    
    def load_existing_ids(self, filepath: str):
        """
        Load existing incident IDs from output file for idempotency
        
        This ensures that running the scraper multiple times per day
        does not create duplicate records.
        
        Args:
            filepath: Path to the output file (CSV or Excel)
        """
        if not os.path.exists(filepath):
            logging.info(f"[Deduplication] No existing file found: {filepath}")
            return
        
        try:
            # Determine file type and load accordingly
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                logging.warning(f"[Deduplication] Unsupported file format: {filepath}")
                return
            
            # Extract incident IDs
            if 'incident_id' in df.columns:
                existing_ids = set(df['incident_id'].dropna().astype(str))
                self.seen_ids.update(existing_ids)
                
                logging.info(
                    f"[Deduplication] Loaded {len(existing_ids)} existing IDs from {filepath}"
                )
                
                # Also load recent incidents for similarity comparison
                # (only incidents from the last 48 hours)
                if len(df) > 0:
                    cutoff_time = datetime.now() - timedelta(hours=self.temporal_window_hours)
                    
                    # Filter recent incidents
                    recent_df = df[df['incident_date'].notna()].copy()
                    
                    for _, row in recent_df.iterrows():
                        try:
                            # Parse incident date
                            incident_date = row['incident_date']
                            if isinstance(incident_date, str):
                                if 'T' in incident_date:
                                    inc_time = datetime.fromisoformat(incident_date)
                                else:
                                    inc_time = datetime.strptime(incident_date, '%Y-%m-%d')
                            else:
                                # pandas Timestamp
                                inc_time = pd.to_datetime(incident_date).to_pydatetime()
                            
                            # Only add if within temporal window
                            if inc_time >= cutoff_time:
                                # Create CrimeIncident from row
                                incident = CrimeIncident(
                                    incident_id=str(row.get('incident_id', '')),
                                    source=str(row.get('source', '')),
                                    source_url=str(row.get('source_url', '')),
                                    scraped_at=str(row.get('scraped_at', '')),
                                    incident_date=str(row.get('incident_date', '')),
                                    incident_time=str(row.get('incident_time', '')),
                                    area=str(row.get('area', '')),
                                    sub_area=str(row.get('sub_area', '')),
                                    location=str(row.get('location', '')),
                                    city=str(row.get('city', 'Karachi')),
                                    location_for_geocoding=str(row.get('location_for_geocoding', '')),
                                    latitude=float(row.get('latitude')) if pd.notna(row.get('latitude')) else None,
                                    longitude=float(row.get('longitude')) if pd.notna(row.get('longitude')) else None,
                                    description=str(row.get('description', '')),
                                    incident_type=str(row.get('incident_type', '')),
                                    device_model=str(row.get('device_model', '')),
                                    victim_phone=str(row.get('victim_phone', '')),
                                    imei_number=str(row.get('imei_number', '')),
                                    confidence_score=float(row.get('confidence_score', 0.0)),
                                    quality_score=float(row.get('quality_score', 0.0)),
                                    is_statistical=bool(row.get('is_statistical', False)),
                                    is_urdu_translated=bool(row.get('is_urdu_translated', False)),
                                    duplicate_sources=eval(str(row.get('duplicate_sources', '[]'))) if pd.notna(row.get('duplicate_sources')) else [],
                                    raw_text=str(row.get('raw_text', '')),
                                    post_timestamp=str(row.get('post_timestamp', ''))
                                )
                                
                                self.recent_incidents.append(incident)
                        
                        except Exception as e:
                            logging.debug(f"[Deduplication] Error loading incident for comparison: {e}")
                            continue
                    
                    logging.info(
                        f"[Deduplication] Loaded {len(self.recent_incidents)} recent incidents "
                        f"for similarity comparison"
                    )
            else:
                logging.warning(f"[Deduplication] No 'incident_id' column found in {filepath}")
        
        except Exception as e:
            logging.error(f"[Deduplication] Error loading existing IDs: {e}")
            logging.debug(f"[Deduplication] Traceback: {traceback.format_exc()}")
    
    def get_stats(self) -> Dict:
        """
        Get deduplication statistics
        
        Returns:
            Dictionary with deduplication stats
        """
        return {
            'total_ids': len(self.seen_ids),
            'recent_incidents': len(self.recent_incidents),
            'similarity_threshold': self.similarity_threshold,
            'temporal_window_hours': self.temporal_window_hours,
            'cache_size': len(self.similarity_cache)
        }


# ============================================================================
# TASK 18: INCIDENT STORE WITH IDEMPOTENT OPERATIONS
# ============================================================================

class IncidentStore:
    """
    Manages persistent storage of crime incidents with idempotent operations.
    
    Features:
    - Excel/CSV output with proper formatting
    - Append-only mode to prevent duplicates
    - Atomic file operations with temporary files
    - Backup mechanism for data safety
    - Idempotent operations (safe to run multiple times)
    - Loads existing incidents to prevent duplicates
    
    Requirements: 2, 22, 29
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize incident store
        
        Args:
            output_dir: Directory for output files (default: 'output')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Backup directory
        self.backup_dir = self.output_dir / 'archive'
        self.backup_dir.mkdir(exist_ok=True)
        
        # Main output files
        self.excel_file = self.output_dir / 'karachi_crimes.xlsx'
        self.csv_file = self.output_dir / 'karachi_crimes.csv'
        
        # Track existing incident IDs for idempotency
        self.existing_ids: set = set()
        
        logging.info(f"[IncidentStore] Initialized with output directory: {self.output_dir.absolute()}")
    
    def load_existing_incidents(self) -> List[CrimeIncident]:
        """
        Load existing incidents from output file to prevent duplicates.
        Implements idempotency by tracking already-saved incident IDs.
        
        Returns:
            List of existing CrimeIncident objects
            
        Requirements: 2, 22
        """
        existing_incidents = []
        
        try:
            # Try to load from Excel first, fallback to CSV
            if self.excel_file.exists():
                logging.info(f"[IncidentStore] Loading existing incidents from {self.excel_file}")
                df = pd.read_excel(self.excel_file, sheet_name='All Incidents')
                source_file = self.excel_file
            elif self.csv_file.exists():
                logging.info(f"[IncidentStore] Loading existing incidents from {self.csv_file}")
                df = pd.read_csv(self.csv_file)
                source_file = self.csv_file
            else:
                logging.info("[IncidentStore] No existing output files found - starting fresh")
                return existing_incidents
            
            # Extract incident IDs for deduplication
            if 'incident_id' in df.columns:
                self.existing_ids = set(df['incident_id'].astype(str).tolist())
                logging.info(f"[IncidentStore] Loaded {len(self.existing_ids)} existing incident IDs")
            else:
                logging.warning("[IncidentStore] No 'incident_id' column found in existing file")
            
            # Convert DataFrame rows to CrimeIncident objects
            for _, row in df.iterrows():
                try:
                    incident = CrimeIncident(
                        incident_id=str(row.get('incident_id', '')),
                        source=str(row.get('source', '')),
                        source_url=str(row.get('source_url', '')),
                        scraped_at=str(row.get('scraped_at', '')),
                        incident_date=str(row.get('incident_date', '')),
                        incident_time=str(row.get('incident_time', '')),
                        area=str(row.get('area', '')),
                        sub_area=str(row.get('sub_area', '')),
                        location=str(row.get('location', '')),
                        city=str(row.get('city', 'Karachi')),
                        location_for_geocoding=str(row.get('location_for_geocoding', '')),
                        latitude=float(row.get('latitude')) if pd.notna(row.get('latitude')) else None,
                        longitude=float(row.get('longitude')) if pd.notna(row.get('longitude')) else None,
                        description=str(row.get('description', '')),
                        incident_type=str(row.get('incident_type', '')),
                        device_model=str(row.get('device_model', '')),
                        victim_phone=str(row.get('victim_phone', '')),
                        imei_number=str(row.get('imei_number', '')),
                        confidence_score=float(row.get('confidence_score', 0.0)),
                        quality_score=float(row.get('quality_score', 0.0)),
                        is_statistical=bool(row.get('is_statistical', False)),
                        is_urdu_translated=bool(row.get('is_urdu_translated', False)),
                        duplicate_sources=eval(str(row.get('duplicate_sources', '[]'))) if pd.notna(row.get('duplicate_sources')) else [],
                        raw_text=str(row.get('raw_text', '')),
                        post_timestamp=str(row.get('post_timestamp', ''))
                    )
                    existing_incidents.append(incident)
                except Exception as e:
                    logging.debug(f"[IncidentStore] Error loading incident row: {e}")
                    continue
            
            logging.info(f"[IncidentStore] Successfully loaded {len(existing_incidents)} existing incidents")
            
        except Exception as e:
            logging.error(f"[IncidentStore] Error loading existing incidents: {e}")
            logging.debug(f"[IncidentStore] Traceback: {traceback.format_exc()}")
        
        return existing_incidents
    
    def save_incidents(self, new_incidents: List[CrimeIncident], append_mode: bool = True) -> bool:
        """
        Save incidents with append-only mode to prevent duplicates.
        Uses atomic file operations with temporary files for data safety.
        
        Args:
            new_incidents: List of new CrimeIncident objects to save
            append_mode: If True, append to existing file; if False, overwrite (default: True)
            
        Returns:
            True if save successful, False otherwise
            
        Requirements: 2, 22, 29
        """
        if not new_incidents:
            logging.warning("[IncidentStore] No incidents to save")
            return False
        
        try:
            # Filter out duplicates if in append mode
            if append_mode:
                # Load existing IDs if not already loaded
                if not self.existing_ids:
                    self.load_existing_incidents()
                
                # Filter out incidents that already exist
                original_count = len(new_incidents)
                new_incidents = [
                    inc for inc in new_incidents 
                    if inc.incident_id not in self.existing_ids
                ]
                
                duplicates_filtered = original_count - len(new_incidents)
                if duplicates_filtered > 0:
                    logging.info(
                        f"[IncidentStore] Filtered {duplicates_filtered} duplicate incidents "
                        f"({len(new_incidents)} new incidents to save)"
                    )
                
                if not new_incidents:
                    logging.info("[IncidentStore] No new incidents to save (all were duplicates)")
                    return True
            
            # Convert incidents to DataFrame
            new_df = pd.DataFrame([inc.to_dict() for inc in new_incidents])
            
            # Load existing data if appending
            if append_mode and (self.excel_file.exists() or self.csv_file.exists()):
                existing_df = None
                
                if self.excel_file.exists():
                    existing_df = pd.read_excel(self.excel_file, sheet_name='All Incidents')
                elif self.csv_file.exists():
                    existing_df = pd.read_csv(self.csv_file)
                
                if existing_df is not None:
                    # Combine existing and new data
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    logging.info(
                        f"[IncidentStore] Appending {len(new_df)} new incidents to "
                        f"{len(existing_df)} existing incidents"
                    )
                else:
                    combined_df = new_df
            else:
                combined_df = new_df
            
            # Create backup before saving (if file exists)
            if self.excel_file.exists():
                self._create_backup(self.excel_file)
            
            # Save to Excel with atomic operation
            self._save_excel_atomic(combined_df)
            
            # Save to CSV backup with atomic operation
            self._save_csv_atomic(combined_df)
            
            # Save to JSON with atomic operation
            self._save_json_atomic(combined_df)
            
            # Update existing IDs cache
            self.existing_ids.update(inc.incident_id for inc in new_incidents)
            
            logging.info(
                f"[IncidentStore] Successfully saved {len(new_incidents)} new incidents "
                f"(total: {len(combined_df)})"
            )
            
            return True
            
        except Exception as e:
            logging.error(f"[IncidentStore] Error saving incidents: {e}")
            logging.error(f"[IncidentStore] Traceback: {traceback.format_exc()}")
            return False
    
    def _save_excel_atomic(self, df: pd.DataFrame):
        """
        Save DataFrame to Excel using atomic file operations.
        Uses temporary file and rename to ensure data integrity.
        
        Args:
            df: DataFrame to save
            
        Requirements: 2, 29
        """
        # Create temporary file
        temp_file = self.output_dir / f'.temp_{self.excel_file.name}'
        
        try:
            # Write to temporary file
            with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                # Main incidents sheet
                df.to_excel(writer, sheet_name='All Incidents', index=False)
                worksheet = writer.sheets['All Incidents']
                
                # Auto-adjust column widths
                for idx, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).apply(len).max(), len(col))
                    # Excel column letters (A, B, C, ...)
                    col_letter = self._get_excel_column_letter(idx)
                    worksheet.column_dimensions[col_letter].width = min(max_len + 2, 50)
                
                # Create summary sheet
                self._create_summary_sheet(writer, df)
                
                # Create area breakdown sheet
                self._create_area_breakdown_sheet(writer, df)
                
                # Create source breakdown sheet
                self._create_source_breakdown_sheet(writer, df)
            
            # Atomic rename (replace old file with new file)
            if self.excel_file.exists():
                self.excel_file.unlink()
            temp_file.rename(self.excel_file)
            
            logging.info(f"[IncidentStore] Excel file saved: {self.excel_file}")
            
        except Exception as e:
            # Clean up temporary file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _save_csv_atomic(self, df: pd.DataFrame):
        """
        Save DataFrame to CSV using atomic file operations.
        Uses temporary file and rename to ensure data integrity.
        
        Args:
            df: DataFrame to save
            
        Requirements: 2, 29
        """
        # Create temporary file
        temp_file = self.output_dir / f'.temp_{self.csv_file.name}'
        
        try:
            # Write to temporary file
            df.to_csv(temp_file, index=False, encoding='utf-8')
            
            # Atomic rename (replace old file with new file)
            if self.csv_file.exists():
                self.csv_file.unlink()
            temp_file.rename(self.csv_file)
            
            logging.info(f"[IncidentStore] CSV backup saved: {self.csv_file}")
            
        except Exception as e:
            # Clean up temporary file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _save_json_atomic(self, df: pd.DataFrame):
        """
        Save DataFrame to JSON using atomic file operations.
        Uses temporary file and rename to ensure data integrity.
        JSON format is ideal for API integration and database imports.
        
        Args:
            df: DataFrame to save
            
        Requirements: Database integration, API compatibility
        """
        json_file = self.output_dir / 'karachi_crimes.json'
        temp_file = self.output_dir / f'.temp_{json_file.name}'
        
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient='records')
            
            # Create structured JSON with metadata
            json_data = {
                'metadata': {
                    'total_incidents': len(records),
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'source': 'Karachi Crime Scraper',
                    'description': 'Crime incidents in Karachi with geocoded locations'
                },
                'incidents': records
            }
            
            # Write to temporary file with pretty formatting
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Atomic rename (replace old file with new file)
            if json_file.exists():
                json_file.unlink()
            temp_file.rename(json_file)
            
            logging.info(f"[IncidentStore] JSON file saved: {json_file} ({len(records)} incidents)")
            
        except Exception as e:
            # Clean up temporary file on error
            if temp_file.exists():
                temp_file.unlink()
            logging.error(f"[IncidentStore] Error saving JSON: {e}")
            raise e
    
    def _create_backup(self, file_path: Path):
        """
        Create backup of existing file before overwriting.
        Implements data safety mechanism.
        
        Args:
            file_path: Path to file to backup
            
        Requirements: 2
        """
        try:
            if not file_path.exists():
                return
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            # Copy file to backup directory
            import shutil
            shutil.copy2(file_path, backup_path)
            
            logging.info(f"[IncidentStore] Backup created: {backup_path}")
            
            # Clean up old backups (keep only last 10)
            self._cleanup_old_backups()
            
        except Exception as e:
            logging.warning(f"[IncidentStore] Failed to create backup: {e}")
    
    def _cleanup_old_backups(self, keep_count: int = 10):
        """
        Clean up old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent backups to keep (default: 10)
        """
        try:
            # Get all backup files
            backup_files = sorted(
                self.backup_dir.glob('*_backup_*'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            # Delete old backups
            for old_backup in backup_files[keep_count:]:
                old_backup.unlink()
                logging.debug(f"[IncidentStore] Deleted old backup: {old_backup.name}")
            
        except Exception as e:
            logging.warning(f"[IncidentStore] Error cleaning up old backups: {e}")
    
    def _get_excel_column_letter(self, idx: int) -> str:
        """
        Convert column index to Excel column letter (0->A, 1->B, ..., 26->AA, etc.)
        
        Args:
            idx: Column index (0-based)
            
        Returns:
            Excel column letter
        """
        result = ""
        while idx >= 0:
            result = chr(65 + (idx % 26)) + result
            idx = idx // 26 - 1
        return result
    
    def _create_summary_sheet(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """
        Create summary statistics sheet in Excel workbook.
        
        Args:
            writer: ExcelWriter object
            df: DataFrame with incident data
        """
        summary_data = {
            'Metric': [
                'Total Incidents',
                'Unique Areas',
                'Avg Quality Score',
                'Avg Confidence Score',
                'With Specific Time',
                'With Device Info',
                'With Phone Number',
                'With IMEI',
                'Statistical Incidents',
                'Urdu Translated'
            ],
            'Value': [
                len(df),
                df['area'].nunique(),
                f"{df['quality_score'].mean():.3f}",
                f"{df['confidence_score'].mean():.3f}",
                (df['incident_time'] != 'Not specified').sum(),
                (df['device_model'] != '').sum(),
                (df['victim_phone'] != '').sum(),
                (df['imei_number'] != '').sum(),
                df['is_statistical'].sum(),
                df['is_urdu_translated'].sum()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    def _create_area_breakdown_sheet(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """
        Create area breakdown sheet in Excel workbook.
        
        Args:
            writer: ExcelWriter object
            df: DataFrame with incident data
        """
        area_stats = df.groupby('area').agg({
            'incident_id': 'count',
            'quality_score': 'mean'
        }).reset_index()
        area_stats.columns = ['Area', 'Count', 'Avg Quality']
        area_stats = area_stats.sort_values('Count', ascending=False)
        area_stats.to_excel(writer, sheet_name='Area Breakdown', index=False)
    
    def _create_source_breakdown_sheet(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """
        Create source breakdown sheet in Excel workbook.
        
        Args:
            writer: ExcelWriter object
            df: DataFrame with incident data
        """
        source_stats = df.groupby('source').agg({
            'incident_id': 'count',
            'quality_score': 'mean'
        }).reset_index()
        source_stats.columns = ['Source', 'Count', 'Avg Quality']
        source_stats = source_stats.sort_values('Count', ascending=False)
        source_stats.to_excel(writer, sheet_name='Source Breakdown', index=False)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the incident store.
        
        Returns:
            Dictionary with store statistics
        """
        stats = {
            'output_dir': str(self.output_dir.absolute()),
            'excel_file': str(self.excel_file),
            'csv_file': str(self.csv_file),
            'excel_exists': self.excel_file.exists(),
            'csv_exists': self.csv_file.exists(),
            'existing_ids_count': len(self.existing_ids),
            'backup_dir': str(self.backup_dir),
            'backup_count': len(list(self.backup_dir.glob('*_backup_*')))
        }
        
        # Add file sizes if files exist
        if self.excel_file.exists():
            stats['excel_size_mb'] = self.excel_file.stat().st_size / (1024 * 1024)
        
        if self.csv_file.exists():
            stats['csv_size_mb'] = self.csv_file.stat().st_size / (1024 * 1024)
        
        return stats


# ============================================================================
# TASK 9: STATISTICAL INCIDENT EXPANDER
# ============================================================================

@dataclass
class StatisticalReport:
    """
    Data model for statistical crime reports
    Represents aggregate crime data that needs to be expanded into individual incidents
    """
    count: int                          # Number of incidents reported
    location: str                       # Location/area mentioned
    time_period: str                    # Time period (e.g., "24 hours", "week", "month")
    incident_type: str                  # Type of crime
    source: str                         # Source name
    source_url: str                     # Source URL
    raw_text: str                       # Original text
    post_timestamp: str                 # When the report was posted
    base_date: str                      # Base date for timestamp distribution


class StatisticalExpander:
    """
    Expand aggregate statistics into individual incident records.
    
    Features:
    - Detect statistical crime reports using regex patterns
    - Verify source credibility before expansion
    - Generate individual incident records from aggregate data
    - Distribute timestamps intelligently across time periods
    - Flag statistical incidents for transparency
    
    Requirements addressed: 18, 20
    """
    
    def __init__(self, knowledge_base: Optional[Dict] = None):
        """
        Initialize statistical expander with verified sources list
        
        Args:
            knowledge_base: Optional knowledge base dictionary containing verified sources
        """
        # Verified sources for statistical data (Requirement 20)
        self.verified_sources = [
            'dawn.com',
            'geo.tv',
            'tribune.com.pk',
            'express.com.pk',
            'karachipoliceonline',
            'sindhpolice.gov.pk',
            'thenews.com.pk',
            'samaa.tv',
            'ary.digital',
            'dunyanews.tv'
        ]
        
        # Load additional verified sources from knowledge base if provided
        if knowledge_base and 'verified_sources' in knowledge_base:
            additional_sources = knowledge_base['verified_sources']
            for source in additional_sources:
                if source not in self.verified_sources:
                    self.verified_sources.append(source)
        
        # Regex patterns for detecting statistical reports (Requirement 18)
        self.statistical_patterns = [
            # Pattern: "250 incidents near Millennium Mall"
            r'(\d+)\s+(?:incidents?|cases?|reports?|crimes?)\s+(?:near|in|at|around)\s+([A-Za-z\s]+)',
            
            # Pattern: "250 mobile snatching incidents in DHA"
            r'(\d+)\s+(?:mobile|phone|cell)?\s*(?:snatching|theft|robbery|stolen)\s+(?:incidents?|cases?|reports?)\s+(?:in|at|near|around)\s+([A-Za-z\s]+)',
            
            # Pattern: "DHA reported 250 incidents"
            r'([A-Za-z\s]+)\s+(?:reported|recorded|witnessed)\s+(\d+)\s+(?:incidents?|cases?|crimes?)',
            
            # Pattern: "250 incidents reported in the last 24 hours"
            r'(\d+)\s+(?:incidents?|cases?|reports?)\s+(?:reported|recorded|occurred)\s+(?:in|during|over)\s+(?:the\s+)?(?:last|past)\s+([0-9]+)\s+(hours?|days?|weeks?|months?)',
            
            # Pattern: "Over 250 mobile phones snatched in Karachi"
            r'(?:over|more than|around|approximately)\s+(\d+)\s+(?:mobile|phone|cell)?\s*(?:phones?|mobiles?)?\s+(?:snatched|stolen|robbed)\s+(?:in|at|near)\s+([A-Za-z\s]+)',
            
            # Pattern: "Police recorded 250 snatching cases"
            r'(?:police|authorities)\s+(?:recorded|reported|registered)\s+(\d+)\s+(?:snatching|theft|robbery)\s+(?:cases?|incidents?)',
            
            # Pattern: "250 street crimes in Karachi this week"
            r'(\d+)\s+(?:street\s+)?(?:crimes?|incidents?)\s+(?:in|at)\s+([A-Za-z\s]+)\s+(?:this|last)\s+(week|month|day)',
        ]
        
        # Time period mappings (for timestamp distribution)
        self.time_period_hours = {
            'hour': 1,
            'hours': 1,
            'day': 24,
            'days': 24,
            'week': 168,  # 7 days
            'weeks': 168,
            'month': 720,  # 30 days
            'months': 720
        }
        
        logging.info(
            f"[StatisticalExpander] Initialized with {len(self.verified_sources)} verified sources"
        )
    
    def detect_statistics(self, text: str) -> Optional[StatisticalReport]:
        """
        Detect if text contains aggregate crime statistics using regex patterns
        
        Args:
            text: Text to analyze
            
        Returns:
            StatisticalReport if statistics detected, None otherwise
        """
        if not text or len(text) < 20:
            return None
        
        text_lower = text.lower()
        
        # Check if text contains statistical indicators
        statistical_keywords = [
            'incidents', 'cases', 'reported', 'recorded', 'registered',
            'statistics', 'data', 'numbers', 'total', 'count'
        ]
        
        has_statistical_keyword = any(keyword in text_lower for keyword in statistical_keywords)
        
        if not has_statistical_keyword:
            return None
        
        # Try each regex pattern
        for pattern in self.statistical_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                groups = match.groups()
                
                # Extract count and location based on pattern
                count = None
                location = "Karachi"
                time_period = "24 hours"  # Default
                
                # Parse matched groups
                for group in groups:
                    if group and group.strip():
                        # Check if it's a number
                        if group.isdigit():
                            if count is None:
                                count = int(group)
                            else:
                                # This might be a time period number
                                time_period = f"{group} hours"
                        # Check if it's a time unit
                        elif group.lower() in ['hour', 'hours', 'day', 'days', 'week', 'weeks', 'month', 'months']:
                            # Find the number before this unit
                            for prev_group in groups:
                                if prev_group and prev_group.isdigit():
                                    time_period = f"{prev_group} {group}"
                                    break
                        # Otherwise, it's likely a location
                        elif len(group) > 2 and not group.isdigit():
                            location = group.strip()
                
                # Validate count
                if count and count > 1:  # Must be at least 2 incidents to be statistical
                    # Determine incident type from text
                    incident_type = "snatching"  # Default
                    if 'theft' in text_lower or 'stolen' in text_lower:
                        incident_type = "theft"
                    elif 'robbery' in text_lower or 'robbed' in text_lower:
                        incident_type = "robbery"
                    elif 'snatch' in text_lower:
                        incident_type = "snatching"
                    
                    logging.info(
                        f"[StatisticalExpander] Detected statistics: {count} {incident_type} "
                        f"incidents in {location} over {time_period}"
                    )
                    
                    return StatisticalReport(
                        count=count,
                        location=location,
                        time_period=time_period,
                        incident_type=incident_type,
                        source="",  # To be filled by caller
                        source_url="",  # To be filled by caller
                        raw_text=text,
                        post_timestamp="",  # To be filled by caller
                        base_date=""  # To be filled by caller
                    )
        
        # Check for simple numeric patterns as fallback
        # Pattern: "250 snatching" or "250 mobile theft"
        simple_pattern = r'(\d+)\s+(?:mobile|phone|cell)?\s*(?:snatching|theft|robbery|stolen|snatch)'
        match = re.search(simple_pattern, text, re.IGNORECASE)
        
        if match:
            count = int(match.group(1))
            
            if count > 1:
                incident_type = "snatching"
                if 'theft' in text_lower:
                    incident_type = "theft"
                elif 'robbery' in text_lower:
                    incident_type = "robbery"
                
                logging.info(
                    f"[StatisticalExpander] Detected statistics (simple pattern): "
                    f"{count} {incident_type} incidents"
                )
                
                return StatisticalReport(
                    count=count,
                    location="Karachi",
                    time_period="24 hours",
                    incident_type=incident_type,
                    source="",
                    source_url="",
                    raw_text=text,
                    post_timestamp="",
                    base_date=""
                )
        
        return None
    
    def verify_source(self, source_url: str) -> bool:
        """
        Verify if source is credible for statistical data
        
        Only verified sources (official police reports, verified news channels)
        should be used for expanding statistics into individual records.
        
        Args:
            source_url: URL of the source
            
        Returns:
            True if source is verified, False otherwise
        """
        if not source_url:
            return False
        
        source_url_lower = source_url.lower()
        
        # Check if URL contains any verified source domain
        for verified_source in self.verified_sources:
            if verified_source.lower() in source_url_lower:
                logging.info(
                    f"[StatisticalExpander] Source verified: {verified_source} in {source_url}"
                )
                return True
        
        logging.debug(
            f"[StatisticalExpander] Source not verified: {source_url}"
        )
        return False
    
    def expand_to_incidents(
        self,
        stats: StatisticalReport,
        source: str,
        source_url: str,
        post_timestamp: str
    ) -> List[CrimeIncident]:
        """
        Generate individual incident records from statistical data
        
        Creates multiple CrimeIncident objects equal to the reported count,
        with timestamps distributed across the reported time period.
        
        Args:
            stats: StatisticalReport containing aggregate data
            source: Source name
            source_url: Source URL
            post_timestamp: When the report was posted
            
        Returns:
            List of CrimeIncident objects
        """
        # Verify source credibility first (Requirement 20)
        if not self.verify_source(source_url):
            logging.warning(
                f"[StatisticalExpander] Source not verified, treating as single incident: {source_url}"
            )
            # Return single incident with aggregated information
            return self._create_single_aggregate_incident(stats, source, source_url, post_timestamp)
        
        logging.info(
            f"[StatisticalExpander] Expanding {stats.count} incidents from verified source"
        )
        
        # Generate timestamps distributed across time period
        timestamps = self.distribute_timestamps(
            count=stats.count,
            time_period=stats.time_period,
            base_timestamp=post_timestamp
        )
        
        incidents = []
        
        # Create individual incident records
        for i, timestamp in enumerate(timestamps):
            # Parse timestamp into date and time
            try:
                dt = datetime.fromisoformat(timestamp)
                incident_date = dt.strftime('%Y-%m-%d')
                incident_time = dt.strftime('%I:%M %p')
            except:
                incident_date = datetime.now().strftime('%Y-%m-%d')
                incident_time = datetime.now().strftime('%I:%M %p')
            
            # Create incident
            incident = CrimeIncident(
                incident_id="",  # Will be generated by deduplication system
                source=source,
                source_url=source_url,
                scraped_at=datetime.now().isoformat(),
                incident_date=incident_date,
                incident_time=incident_time,
                area=stats.location if stats.location != "Karachi" else "Unknown",
                sub_area="",
                location=stats.location,
                city="Karachi",
                location_for_geocoding=f"{stats.location}, Karachi, Pakistan",
                latitude=None,  # Will be geocoded later
                longitude=None,  # Will be geocoded later
                description=f"Statistical incident #{i+1} of {stats.count}: {stats.incident_type} reported in {stats.location} during {stats.time_period}",
                incident_type=stats.incident_type,
                device_model="",
                victim_phone="",
                imei_number="",
                confidence_score=0.6,  # Lower confidence for statistical data
                quality_score=0.5,  # Lower quality for generated data
                is_statistical=True,  # Flag as statistical (Requirement 18)
                is_urdu_translated=False,
                duplicate_sources=[],
                raw_text=stats.raw_text,
                post_timestamp=post_timestamp
            )
            
            incidents.append(incident)
        
        logging.info(
            f"[StatisticalExpander] Generated {len(incidents)} individual incidents from statistics"
        )
        
        return incidents
    
    def _create_single_aggregate_incident(
        self,
        stats: StatisticalReport,
        source: str,
        source_url: str,
        post_timestamp: str
    ) -> List[CrimeIncident]:
        """
        Create a single incident with aggregated information for unverified sources
        
        Args:
            stats: StatisticalReport
            source: Source name
            source_url: Source URL
            post_timestamp: Post timestamp
            
        Returns:
            List containing single CrimeIncident
        """
        try:
            dt = datetime.fromisoformat(post_timestamp) if post_timestamp else datetime.now()
        except:
            dt = datetime.now()
        
        incident = CrimeIncident(
            incident_id="",
            source=source,
            source_url=source_url,
            scraped_at=datetime.now().isoformat(),
            incident_date=dt.strftime('%Y-%m-%d'),
            incident_time=dt.strftime('%I:%M %p'),
            area=stats.location if stats.location != "Karachi" else "Unknown",
            sub_area="",
            location=stats.location,
            city="Karachi",
            location_for_geocoding=f"{stats.location}, Karachi, Pakistan",
            latitude=None,  # Will be geocoded later
            longitude=None,  # Will be geocoded later
            description=f"Aggregate report: {stats.count} {stats.incident_type} incidents in {stats.location} during {stats.time_period}",
            incident_type=stats.incident_type,
            device_model="",
            victim_phone="",
            imei_number="",
            confidence_score=0.4,  # Low confidence for unverified aggregate
            quality_score=0.4,
            is_statistical=True,
            is_urdu_translated=False,
            duplicate_sources=[],
            raw_text=stats.raw_text,
            post_timestamp=post_timestamp
        )
        
        return [incident]
    
    def distribute_timestamps(
        self,
        count: int,
        time_period: str,
        base_timestamp: str
    ) -> List[str]:
        """
        Distribute timestamps across the reported time period
        
        Spreads incident timestamps evenly across the time period to simulate
        realistic temporal distribution.
        
        Args:
            count: Number of incidents to generate timestamps for
            time_period: Time period string (e.g., "24 hours", "1 week")
            base_timestamp: Base timestamp (usually post timestamp)
            
        Returns:
            List of ISO timestamp strings
        """
        # Parse base timestamp
        try:
            if base_timestamp:
                base_dt = datetime.fromisoformat(base_timestamp)
            else:
                base_dt = datetime.now()
        except:
            base_dt = datetime.now()
        
        # Parse time period to get hours
        period_hours = 24  # Default to 24 hours
        
        # Extract number and unit from time period
        period_match = re.search(r'(\d+)\s*(hour|day|week|month)', time_period.lower())
        
        if period_match:
            number = int(period_match.group(1))
            unit = period_match.group(2)
            
            if unit in self.time_period_hours:
                period_hours = number * self.time_period_hours[unit]
        else:
            # Try to find just the unit
            for unit, hours in self.time_period_hours.items():
                if unit in time_period.lower():
                    period_hours = hours
                    break
        
        logging.debug(
            f"[StatisticalExpander] Distributing {count} timestamps over {period_hours} hours"
        )
        
        # Calculate time interval between incidents
        if count <= 1:
            return [base_dt.isoformat()]
        
        # Distribute evenly with some randomness
        timestamps = []
        interval_hours = period_hours / count
        
        for i in range(count):
            # Calculate offset with some randomness (±20% of interval)
            base_offset = i * interval_hours
            randomness = interval_hours * 0.2 * (random.random() - 0.5)
            offset_hours = base_offset + randomness
            
            # Ensure offset is within bounds
            offset_hours = max(0, min(offset_hours, period_hours))
            
            # Calculate timestamp
            incident_dt = base_dt - timedelta(hours=offset_hours)
            timestamps.append(incident_dt.isoformat())
        
        # Shuffle to avoid sequential pattern
        random.shuffle(timestamps)
        
        logging.debug(
            f"[StatisticalExpander] Generated {len(timestamps)} timestamps "
            f"(range: {period_hours} hours)"
        )
        
        return timestamps
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the statistical expander
        
        Returns:
            Dictionary with expander stats
        """
        return {
            'verified_sources_count': len(self.verified_sources),
            'verified_sources': self.verified_sources,
            'pattern_count': len(self.statistical_patterns)
        }


# ============================================================================
# TASK 10: FACEBOOK SCRAPING ENGINE WITH MULTIPLE SELECTOR STRATEGIES
# ============================================================================

class FacebookEngine:
    """
    Facebook scraping engine with multiple selector strategies and automatic fallback.
    
    Features:
    - Multiple CSS selector strategies with automatic fallback
    - Scroll-based infinite loading for comprehensive data collection
    - Post data extraction (text, author, metadata)
    - Post timestamp extraction from HTML attributes
    - Related group discovery from post content
    - Selector effectiveness testing and ranking
    - Anti-detection measures (random delays, human-like scrolling)
    - Integration with driver pool for robust error handling
    
    Requirements addressed: 8, 11, 16, 27
    """
    
    def __init__(self, driver_manager: DriverPoolManager, error_recovery: Optional[ErrorRecoverySystem] = None):
        """
        Initialize Facebook scraping engine
        
        Args:
            driver_manager: DriverPoolManager instance for driver management
            error_recovery: Optional ErrorRecoverySystem for error handling
        """
        self.driver_manager = driver_manager
        self.error_recovery = error_recovery or ErrorRecoverySystem(driver_manager)
        
        # Multiple selector strategies for Facebook posts (ordered by priority)
        # Facebook frequently changes their HTML structure, so we maintain multiple strategies
        self.selector_strategies = [
            # Strategy 1: Modern Facebook (2024+)
            {
                'name': 'modern_2024',
                'post_container': 'div[role="article"]',
                'post_text': 'div[data-ad-comet-preview="message"], div[data-ad-preview="message"]',
                'post_author': 'h2 a, h3 a, h4 a, strong a',
                'post_timestamp': 'a[href*="/posts/"] abbr, a[href*="/permalink/"] abbr, span[id^="jsc_"] abbr',
                'success_count': 0,
                'failure_count': 0
            },
            # Strategy 2: Classic Facebook structure
            {
                'name': 'classic',
                'post_container': 'div.userContentWrapper, div._5pcr',
                'post_text': 'div.userContent, div[data-testid="post_message"]',
                'post_author': 'a.profileLink, h5 a',
                'post_timestamp': 'abbr[data-utime], abbr._5ptz',
                'success_count': 0,
                'failure_count': 0
            },
            # Strategy 3: Mobile Facebook structure
            {
                'name': 'mobile',
                'post_container': 'article, div[data-ft]',
                'post_text': 'div[dir="auto"], p',
                'post_author': 'h3 a, strong a',
                'post_timestamp': 'abbr',
                'success_count': 0,
                'failure_count': 0
            },
            # Strategy 4: Generic fallback
            {
                'name': 'generic_fallback',
                'post_container': 'div[role="article"], article, div[data-ft]',
                'post_text': 'div[dir="auto"], div.userContent, p',
                'post_author': 'a[href*="/user/"], a[href*="/profile/"], h2 a, h3 a, strong a',
                'post_timestamp': 'abbr, time, span[data-utime]',
                'success_count': 0,
                'failure_count': 0
            },
            # Strategy 5: Discovery mode (finds any text content)
            {
                'name': 'discovery',
                'post_container': 'div',
                'post_text': 'div, p, span',
                'post_author': 'a',
                'post_timestamp': 'abbr, time',
                'success_count': 0,
                'failure_count': 0
            }
        ]
        
        # Crime-related keywords for validation
        self.crime_keywords = [
            'snatch', 'snatching', 'snatched', 'چھین', 'چھینا',
            'theft', 'stolen', 'چوری', 'چور',
            'robbery', 'robbed', 'لوٹ', 'ڈکیتی',
            'mobile', 'phone', 'موبائل', 'فون',
            'crime', 'incident', 'جرم', 'واقعہ',
            'mugging', 'mugged', 'لوٹ مار',
            'dacoity', 'ڈکیتی'
        ]
        
        # Karachi area keywords for group discovery
        self.karachi_areas = [
            'dha', 'clifton', 'saddar', 'gulshan', 'nazimabad',
            'north nazimabad', 'malir', 'korangi', 'landhi',
            'orangi', 'liaquatabad', 'shah faisal', 'bin qasim',
            'kemari', 'lyari', 'jamshed', 'garden', 'baldia',
            'site', 'north karachi', 'new karachi', 'federal b area',
            'pechs', 'bahadurabad', 'tariq road', 'shahrah-e-faisal'
        ]
        
        # Statistics tracking
        self.stats = {
            'total_posts_found': 0,
            'valid_posts_extracted': 0,
            'groups_discovered': 0,
            'scroll_operations': 0,
            'selector_switches': 0
        }
        self.stats_lock = threading.Lock()
        
        logging.info(
            f"[FacebookEngine] Initialized with {len(self.selector_strategies)} selector strategies"
        )
    
    def scrape_group(
        self,
        group_url: str,
        max_scrolls: int = 10,
        max_posts: int = 100,
        scroll_delay: float = 3.0
    ) -> List[Dict]:
        """
        Scrape posts from a Facebook group with scroll-based infinite loading
        
        Args:
            group_url: URL of the Facebook group to scrape
            max_scrolls: Maximum number of scroll operations (default: 10)
            max_posts: Maximum number of posts to collect (default: 100)
            scroll_delay: Delay between scrolls in seconds (default: 3.0)
            
        Returns:
            List of extracted post dictionaries
        """
        logging.info(f"[FacebookEngine] Starting to scrape group: {group_url}")
        
        posts = []
        driver = None
        
        try:
            # Get driver from pool
            driver = self.driver_manager.get_driver()
            
            # Navigate to group URL
            logging.info(f"[FacebookEngine] Navigating to {group_url}")
            driver.get(group_url)
            
            # Wait for page to load
            time.sleep(5)  # Initial load time
            
            # Test selectors to find the best strategy
            active_strategy = self._test_selectors(driver)
            
            if not active_strategy:
                logging.warning(f"[FacebookEngine] No working selector strategy found for {group_url}")
                return posts
            
            logging.info(f"[FacebookEngine] Using selector strategy: {active_strategy['name']}")
            
            # Scroll and collect posts
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_count = 0
            posts_seen = set()
            
            while scroll_count < max_scrolls and len(posts) < max_posts:
                # Find post elements using active strategy
                try:
                    post_elements = driver.find_elements(
                        By.CSS_SELECTOR,
                        active_strategy['post_container']
                    )
                    
                    logging.debug(
                        f"[FacebookEngine] Found {len(post_elements)} post elements "
                        f"(scroll {scroll_count + 1}/{max_scrolls})"
                    )
                    
                    # Extract data from new posts
                    for element in post_elements:
                        try:
                            # Generate unique identifier for this element
                            element_id = id(element)
                            
                            if element_id in posts_seen:
                                continue
                            
                            posts_seen.add(element_id)
                            
                            # Extract post data
                            post_data = self.extract_post_data(element, active_strategy, driver)
                            
                            if post_data and self._is_crime_related(post_data.get('text', '')):
                                posts.append(post_data)
                                self._update_stats('valid_posts_extracted')
                                
                                logging.info(
                                    f"[FacebookEngine] Extracted valid post "
                                    f"(total: {len(posts)}/{max_posts})"
                                )
                                
                                # Check if we've reached max posts
                                if len(posts) >= max_posts:
                                    break
                            
                            self._update_stats('total_posts_found')
                            
                        except Exception as e:
                            logging.debug(f"[FacebookEngine] Error extracting post data: {e}")
                            continue
                    
                    # Check if we've collected enough posts
                    if len(posts) >= max_posts:
                        logging.info(f"[FacebookEngine] Reached max posts limit ({max_posts})")
                        break
                    
                except Exception as e:
                    logging.warning(f"[FacebookEngine] Error finding post elements: {e}")
                
                # Perform human-like scrolling
                self._human_like_scroll(driver, scroll_delay)
                scroll_count += 1
                self._update_stats('scroll_operations')
                
                # Check if we've reached the bottom
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logging.info(f"[FacebookEngine] Reached bottom of page")
                    break
                
                last_height = new_height
            
            logging.info(
                f"[FacebookEngine] Scraping completed: {len(posts)} valid posts from {group_url}"
            )
            
        except Exception as e:
            logging.error(f"[FacebookEngine] Error scraping group {group_url}: {e}")
            logging.error(traceback.format_exc())
            
        finally:
            # Return driver to pool
            if driver:
                self.driver_manager.release_driver(driver)
        
        return posts
    
    def extract_post_data(
        self,
        element,
        strategy: Dict,
        driver
    ) -> Optional[Dict]:
        """
        Extract text, author, and metadata from a post element
        
        Args:
            element: WebElement representing the post
            strategy: Selector strategy dictionary to use
            driver: WebDriver instance for additional operations
            
        Returns:
            Dictionary with extracted post data or None if extraction failed
        """
        try:
            post_data = {
                'text': '',
                'author': 'Unknown',
                'timestamp': None,
                'url': '',
                'source': 'Facebook',
                'raw_html': ''
            }
            
            # Extract post text
            try:
                text_elements = element.find_elements(By.CSS_SELECTOR, strategy['post_text'])
                if text_elements:
                    # Get text from all matching elements and join
                    texts = [elem.text.strip() for elem in text_elements if elem.text.strip()]
                    post_data['text'] = ' '.join(texts)
            except Exception as e:
                logging.debug(f"[FacebookEngine] Error extracting text: {e}")
            
            # Skip if no text found
            if not post_data['text'] or len(post_data['text']) < 10:
                return None
            
            # Extract author
            try:
                author_elements = element.find_elements(By.CSS_SELECTOR, strategy['post_author'])
                if author_elements:
                    post_data['author'] = author_elements[0].text.strip() or 'Unknown'
            except Exception as e:
                logging.debug(f"[FacebookEngine] Error extracting author: {e}")
            
            # Extract timestamp
            post_data['timestamp'] = self.get_post_timestamp(element, strategy)
            
            # Extract post URL
            try:
                # Try to find permalink
                link_elements = element.find_elements(
                    By.CSS_SELECTOR,
                    'a[href*="/posts/"], a[href*="/permalink/"], a[href*="/story.php"]'
                )
                if link_elements:
                    post_data['url'] = link_elements[0].get_attribute('href')
            except Exception as e:
                logging.debug(f"[FacebookEngine] Error extracting URL: {e}")
            
            # Store raw HTML for debugging (first 500 chars)
            try:
                post_data['raw_html'] = element.get_attribute('outerHTML')[:500]
            except:
                pass
            
            return post_data
            
        except Exception as e:
            logging.debug(f"[FacebookEngine] Error in extract_post_data: {e}")
            return None
    
    def get_post_timestamp(self, element, strategy: Dict) -> Optional[str]:
        """
        Extract post publication time from HTML attributes
        
        Args:
            element: WebElement representing the post
            strategy: Selector strategy dictionary to use
            
        Returns:
            ISO format timestamp string or None if not found
        """
        try:
            # Try to find timestamp elements
            timestamp_elements = element.find_elements(
                By.CSS_SELECTOR,
                strategy['post_timestamp']
            )
            
            for ts_elem in timestamp_elements:
                # Try data-utime attribute (Unix timestamp)
                utime = ts_elem.get_attribute('data-utime')
                if utime:
                    try:
                        timestamp = datetime.fromtimestamp(int(utime))
                        return timestamp.isoformat()
                    except:
                        pass
                
                # Try datetime attribute
                dt = ts_elem.get_attribute('datetime')
                if dt:
                    return dt
                
                # Try title attribute (often contains full date)
                title = ts_elem.get_attribute('title')
                if title:
                    # Try to parse various date formats
                    try:
                        # Common Facebook format: "Monday, January 15, 2024 at 3:45 PM"
                        timestamp = datetime.strptime(title, "%A, %B %d, %Y at %I:%M %p")
                        return timestamp.isoformat()
                    except:
                        pass
                    
                    try:
                        # Alternative format: "January 15 at 3:45 PM"
                        current_year = datetime.now().year
                        timestamp = datetime.strptime(f"{title} {current_year}", "%B %d at %I:%M %p %Y")
                        return timestamp.isoformat()
                    except:
                        pass
                
                # Try text content as last resort
                text = ts_elem.text.strip()
                if text:
                    # Handle relative times like "2h", "3d", "1w"
                    if 'h' in text or 'hr' in text or 'hour' in text:
                        hours = int(re.search(r'\d+', text).group())
                        timestamp = datetime.now() - timedelta(hours=hours)
                        return timestamp.isoformat()
                    elif 'd' in text or 'day' in text:
                        days = int(re.search(r'\d+', text).group())
                        timestamp = datetime.now() - timedelta(days=days)
                        return timestamp.isoformat()
                    elif 'w' in text or 'week' in text:
                        weeks = int(re.search(r'\d+', text).group())
                        timestamp = datetime.now() - timedelta(weeks=weeks)
                        return timestamp.isoformat()
            
            # If no timestamp found, use current time
            return datetime.now().isoformat()
            
        except Exception as e:
            logging.debug(f"[FacebookEngine] Error extracting timestamp: {e}")
            return datetime.now().isoformat()
    
    def discover_related_groups(self, post_text: str) -> List[str]:
        """
        Find new Facebook group URLs mentioned in post content
        
        Args:
            post_text: Text content of the post
            
        Returns:
            List of discovered Facebook group URLs
        """
        discovered_groups = []
        
        try:
            # Pattern 1: Direct Facebook group URLs
            fb_group_pattern = r'https?://(?:www\.)?facebook\.com/groups/[a-zA-Z0-9._-]+'
            matches = re.findall(fb_group_pattern, post_text)
            discovered_groups.extend(matches)
            
            # Pattern 2: Facebook group IDs (numeric)
            fb_group_id_pattern = r'facebook\.com/groups/(\d+)'
            id_matches = re.findall(fb_group_id_pattern, post_text)
            for group_id in id_matches:
                discovered_groups.append(f"https://www.facebook.com/groups/{group_id}")
            
            # Pattern 3: Mentions of Karachi area groups
            # Look for patterns like "join DHA Crime Alert group" or "Clifton Safety Group"
            for area in self.karachi_areas:
                area_pattern = rf'{area}\s+(?:crime|safety|alert|watch|community)\s+group'
                if re.search(area_pattern, post_text, re.IGNORECASE):
                    logging.info(
                        f"[FacebookEngine] Potential group mention found: {area} group"
                    )
                    # Note: We can't construct the URL without the actual group ID
                    # This is logged for manual investigation
            
            # Remove duplicates
            discovered_groups = list(set(discovered_groups))
            
            if discovered_groups:
                self._update_stats('groups_discovered', len(discovered_groups))
                logging.info(
                    f"[FacebookEngine] Discovered {len(discovered_groups)} group URLs in post"
                )
            
        except Exception as e:
            logging.debug(f"[FacebookEngine] Error discovering groups: {e}")
        
        return discovered_groups
    
    def test_selectors(self, driver) -> Optional[Dict]:
        """
        Test and rank selector effectiveness for the current page
        
        Args:
            driver: WebDriver instance
            
        Returns:
            Best performing selector strategy or None if all fail
        """
        return self._test_selectors(driver)
    
    def _test_selectors(self, driver) -> Optional[Dict]:
        """
        Internal method to test all selector strategies and find the best one
        
        Args:
            driver: WebDriver instance
            
        Returns:
            Best performing selector strategy or None if all fail
        """
        logging.info("[FacebookEngine] Testing selector strategies...")
        
        best_strategy = None
        max_posts_found = 0
        
        for strategy in self.selector_strategies:
            try:
                # Try to find post containers with this strategy
                elements = driver.find_elements(By.CSS_SELECTOR, strategy['post_container'])
                
                if not elements:
                    strategy['failure_count'] += 1
                    logging.debug(
                        f"[FacebookEngine] Strategy '{strategy['name']}' found 0 posts"
                    )
                    continue
                
                # Count how many elements have valid text content
                valid_posts = 0
                for elem in elements[:10]:  # Test first 10 elements
                    try:
                        text_elements = elem.find_elements(By.CSS_SELECTOR, strategy['post_text'])
                        if text_elements:
                            text = ' '.join([te.text.strip() for te in text_elements if te.text.strip()])
                            if len(text) > 20:  # Valid post should have substantial text
                                valid_posts += 1
                    except:
                        continue
                
                logging.info(
                    f"[FacebookEngine] Strategy '{strategy['name']}': "
                    f"found {len(elements)} containers, {valid_posts} with valid text"
                )
                
                if valid_posts > 0:
                    strategy['success_count'] += 1
                    
                    if valid_posts > max_posts_found:
                        max_posts_found = valid_posts
                        best_strategy = strategy
                else:
                    strategy['failure_count'] += 1
                
            except Exception as e:
                strategy['failure_count'] += 1
                logging.debug(f"[FacebookEngine] Strategy '{strategy['name']}' failed: {e}")
        
        if best_strategy:
            logging.info(
                f"[FacebookEngine] Selected strategy: {best_strategy['name']} "
                f"(found {max_posts_found} valid posts)"
            )
            self._update_stats('selector_switches')
        else:
            logging.warning("[FacebookEngine] No working selector strategy found")
        
        return best_strategy
    
    def _human_like_scroll(self, driver, delay: float):
        """
        Perform human-like scrolling with random variations
        
        Args:
            driver: WebDriver instance
            delay: Base delay in seconds
        """
        try:
            # Random scroll distance (70-90% of viewport height)
            scroll_percentage = random.uniform(0.7, 0.9)
            
            # Execute smooth scroll
            driver.execute_script(f"""
                window.scrollBy({{
                    top: window.innerHeight * {scroll_percentage},
                    behavior: 'smooth'
                }});
            """)
            
            # Random delay with variation (±30%)
            actual_delay = delay * random.uniform(0.7, 1.3)
            time.sleep(actual_delay)
            
            # Occasionally scroll back up a bit (10% chance)
            if random.random() < 0.1:
                driver.execute_script("""
                    window.scrollBy({
                        top: -window.innerHeight * 0.2,
                        behavior: 'smooth'
                    });
                """)
                time.sleep(0.5)
            
        except Exception as e:
            logging.debug(f"[FacebookEngine] Error in human-like scroll: {e}")
            # Fallback to simple scroll
            time.sleep(delay)
    
    def _is_crime_related(self, text: str) -> bool:
        """
        Check if text contains crime-related keywords
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be crime-related
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for crime keywords
        for keyword in self.crime_keywords:
            if keyword.lower() in text_lower:
                return True
        
        return False
    
    def _update_stats(self, key: str, increment: int = 1):
        """Update statistics"""
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + increment
    
    def get_stats(self) -> Dict:
        """
        Get scraping statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            stats = dict(self.stats)
            
            # Add selector strategy performance
            stats['selector_strategies'] = [
                {
                    'name': s['name'],
                    'success_count': s['success_count'],
                    'failure_count': s['failure_count'],
                    'success_rate': (
                        s['success_count'] / (s['success_count'] + s['failure_count'])
                        if (s['success_count'] + s['failure_count']) > 0
                        else 0.0
                    )
                }
                for s in self.selector_strategies
            ]
            
            return stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'total_posts_found': 0,
                'valid_posts_extracted': 0,
                'groups_discovered': 0,
                'scroll_operations': 0,
                'selector_switches': 0
            }
            
            # Reset selector strategy stats
            for strategy in self.selector_strategies:
                strategy['success_count'] = 0
                strategy['failure_count'] = 0
            
            logging.info("[FacebookEngine] Statistics reset")


# ============================================================================
# TASK 11: TWITTER SCRAPING ENGINE
# ============================================================================

class TwitterEngine:
    """
    Twitter scraping engine with API and web fallback for crime incident collection.
    
    Features:
    - Keyword-based search for crime-related tweets
    - Thread extraction for full conversation context
    - Rate limit handling with exponential backoff
    - Hashtag and mention tracking
    - Timestamp extraction from tweets
    - Anti-detection measures
    - Integration with driver pool for robust error handling
    
    Requirements addressed: 11, 16, 27
    """
    
    def __init__(self, driver_manager: DriverPoolManager, error_recovery: Optional[ErrorRecoverySystem] = None):
        """
        Initialize Twitter scraping engine
        
        Args:
            driver_manager: DriverPoolManager instance for driver management
            error_recovery: Optional ErrorRecoverySystem for error handling
        """
        self.driver_manager = driver_manager
        self.error_recovery = error_recovery or ErrorRecoverySystem(driver_manager)
        
        # Twitter search queries for Karachi crime
        self.search_queries = [
            'Karachi mobile snatching',
            'Karachi theft',
            'Karachi robbery',
            'Karachi crime',
            'Karachi mugging',
            'DHA snatching',
            'Clifton theft',
            'Gulshan robbery',
            'Karachi phone stolen'
        ]
        
        # Crime-related keywords for validation
        self.crime_keywords = [
            'snatch', 'snatching', 'snatched',
            'theft', 'stolen', 'stole',
            'robbery', 'robbed', 'rob',
            'mobile', 'phone', 'cell',
            'crime', 'incident',
            'mugging', 'mugged',
            'dacoity', 'loot'
        ]
        
        # Karachi area keywords
        self.karachi_areas = [
            'karachi', 'dha', 'clifton', 'saddar', 'gulshan', 'nazimabad',
            'north nazimabad', 'malir', 'korangi', 'landhi',
            'orangi', 'liaquatabad', 'shah faisal', 'bin qasim',
            'kemari', 'lyari', 'jamshed', 'garden', 'baldia',
            'site', 'north karachi', 'new karachi', 'federal b area',
            'pechs', 'bahadurabad', 'tariq road', 'shahrah-e-faisal'
        ]
        
        # CSS selectors for Twitter elements (multiple strategies for robustness)
        self.selectors = {
            'tweet_container': [
                'article[data-testid="tweet"]',
                'div[data-testid="tweet"]',
                'article[role="article"]',
                'div.tweet'
            ],
            'tweet_text': [
                'div[data-testid="tweetText"]',
                'div[lang]',
                'div.tweet-text',
                'span.css-901oao'
            ],
            'tweet_author': [
                'div[data-testid="User-Name"] span',
                'a[role="link"] span',
                'div.tweet-author'
            ],
            'tweet_timestamp': [
                'time',
                'a[href*="/status/"] time',
                'span[data-testid="timestamp"]'
            ],
            'tweet_link': [
                'a[href*="/status/"]',
                'article a[href*="/status/"]'
            ],
            'hashtag': [
                'a[href*="/hashtag/"]',
                'span[data-text="true"]'
            ],
            'mention': [
                'a[href^="/@"]',
                'a[data-testid="mention"]'
            ]
        }
        
        # Rate limiting configuration
        self.rate_limit_config = {
            'requests_per_minute': 15,
            'backoff_base': 5,  # seconds
            'max_backoff': 300,  # 5 minutes
            'max_retries': 3
        }
        
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'total_tweets_found': 0,
            'valid_tweets_extracted': 0,
            'threads_extracted': 0,
            'hashtags_tracked': 0,
            'mentions_tracked': 0,
            'rate_limits_hit': 0,
            'searches_performed': 0
        }
        self.stats_lock = threading.Lock()
        
        logging.info("[TwitterEngine] Initialized with web scraping capabilities")
    
    def scrape_search(
        self,
        query: str,
        max_tweets: int = 50,
        scroll_delay: float = 2.0
    ) -> List[Dict]:
        """
        Search Twitter for keyword-based crime reports
        
        Args:
            query: Search query string
            max_tweets: Maximum number of tweets to collect (default: 50)
            scroll_delay: Delay between scrolls in seconds (default: 2.0)
            
        Returns:
            List of extracted tweet dictionaries
        """
        logging.info(f"[TwitterEngine] Starting search for: {query}")
        
        tweets = []
        driver = None
        
        try:
            # Check rate limiting
            self._check_rate_limit()
            
            # Get driver from pool
            driver = self.driver_manager.get_driver()
            
            # Construct Twitter search URL
            encoded_query = query.replace(' ', '%20')
            search_url = f"https://twitter.com/search?q={encoded_query}&src=typed_query&f=live"
            
            logging.info(f"[TwitterEngine] Navigating to search: {search_url}")
            driver.get(search_url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Scroll and collect tweets
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_count = 0
            max_scrolls = 10
            tweets_seen = set()
            
            while scroll_count < max_scrolls and len(tweets) < max_tweets:
                # Find tweet elements
                tweet_elements = self._find_elements(driver, self.selectors['tweet_container'])
                
                logging.debug(
                    f"[TwitterEngine] Found {len(tweet_elements)} tweet elements "
                    f"(scroll {scroll_count + 1}/{max_scrolls})"
                )
                
                # Extract data from tweets
                for element in tweet_elements:
                    try:
                        # Generate unique identifier
                        element_id = id(element)
                        
                        if element_id in tweets_seen:
                            continue
                        
                        tweets_seen.add(element_id)
                        
                        # Extract tweet data
                        tweet_data = self._extract_tweet_data(element, driver)
                        
                        if tweet_data and self._is_crime_related(tweet_data.get('text', '')):
                            # Extract hashtags and mentions
                            tweet_data['hashtags'] = self._extract_hashtags(element)
                            tweet_data['mentions'] = self._extract_mentions(element)
                            
                            tweets.append(tweet_data)
                            self._update_stats('valid_tweets_extracted')
                            
                            # Track hashtags and mentions
                            self._update_stats('hashtags_tracked', len(tweet_data['hashtags']))
                            self._update_stats('mentions_tracked', len(tweet_data['mentions']))
                            
                            logging.info(
                                f"[TwitterEngine] Extracted valid tweet "
                                f"(total: {len(tweets)}/{max_tweets})"
                            )
                            
                            if len(tweets) >= max_tweets:
                                break
                        
                        self._update_stats('total_tweets_found')
                        
                    except Exception as e:
                        logging.debug(f"[TwitterEngine] Error extracting tweet: {e}")
                        continue
                
                # Check if we've collected enough
                if len(tweets) >= max_tweets:
                    break
                
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_delay)
                scroll_count += 1
                
                # Check if reached bottom
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logging.info("[TwitterEngine] Reached bottom of search results")
                    break
                
                last_height = new_height
            
            self._update_stats('searches_performed')
            logging.info(
                f"[TwitterEngine] Search completed: {len(tweets)} valid tweets for '{query}'"
            )
            
        except Exception as e:
            logging.error(f"[TwitterEngine] Error during search '{query}': {e}")
            logging.error(traceback.format_exc())
            
            # Check if it's a rate limit error
            if 'rate limit' in str(e).lower():
                self._handle_rate_limit()
        
        finally:
            # Return driver to pool
            if driver:
                self.driver_manager.release_driver(driver)
        
        return tweets
    
    def extract_thread(self, tweet_url: str) -> List[Dict]:
        """
        Extract full conversation thread from a tweet URL
        
        Args:
            tweet_url: URL of the tweet to extract thread from
            
        Returns:
            List of tweet dictionaries in the thread (chronological order)
        """
        logging.info(f"[TwitterEngine] Extracting thread from: {tweet_url}")
        
        thread_tweets = []
        driver = None
        
        try:
            # Check rate limiting
            self._check_rate_limit()
            
            # Get driver from pool
            driver = self.driver_manager.get_driver()
            
            # Navigate to tweet
            logging.info(f"[TwitterEngine] Navigating to tweet: {tweet_url}")
            driver.get(tweet_url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Find all tweet elements in the thread
            tweet_elements = self._find_elements(driver, self.selectors['tweet_container'])
            
            logging.info(f"[TwitterEngine] Found {len(tweet_elements)} tweets in thread")
            
            # Extract data from each tweet in thread
            for element in tweet_elements:
                try:
                    tweet_data = self._extract_tweet_data(element, driver)
                    
                    if tweet_data:
                        # Extract hashtags and mentions
                        tweet_data['hashtags'] = self._extract_hashtags(element)
                        tweet_data['mentions'] = self._extract_mentions(element)
                        tweet_data['is_thread'] = True
                        
                        thread_tweets.append(tweet_data)
                        
                        # Track hashtags and mentions
                        self._update_stats('hashtags_tracked', len(tweet_data['hashtags']))
                        self._update_stats('mentions_tracked', len(tweet_data['mentions']))
                    
                except Exception as e:
                    logging.debug(f"[TwitterEngine] Error extracting thread tweet: {e}")
                    continue
            
            self._update_stats('threads_extracted')
            logging.info(
                f"[TwitterEngine] Thread extraction completed: {len(thread_tweets)} tweets"
            )
            
        except Exception as e:
            logging.error(f"[TwitterEngine] Error extracting thread from {tweet_url}: {e}")
            logging.error(traceback.format_exc())
            
            # Check if it's a rate limit error
            if 'rate limit' in str(e).lower():
                self._handle_rate_limit()
        
        finally:
            # Return driver to pool
            if driver:
                self.driver_manager.release_driver(driver)
        
        return thread_tweets
    
    def _extract_tweet_data(self, element, driver) -> Optional[Dict]:
        """
        Extract data from a tweet element
        
        Args:
            element: WebElement representing the tweet
            driver: WebDriver instance
            
        Returns:
            Dictionary with extracted tweet data or None if extraction failed
        """
        try:
            tweet_data = {
                'text': '',
                'author': 'Unknown',
                'timestamp': None,
                'url': '',
                'source': 'Twitter',
                'raw_html': ''
            }
            
            # Extract tweet text
            text_elements = self._find_elements_in_parent(element, self.selectors['tweet_text'])
            if text_elements:
                texts = [elem.text.strip() for elem in text_elements if elem.text.strip()]
                tweet_data['text'] = ' '.join(texts)
            
            if not tweet_data['text']:
                return None
            
            # Extract author
            author_elements = self._find_elements_in_parent(element, self.selectors['tweet_author'])
            if author_elements:
                tweet_data['author'] = author_elements[0].text.strip()
            
            # Extract timestamp
            timestamp = self._extract_timestamp(element)
            tweet_data['timestamp'] = timestamp
            
            # Extract tweet URL
            link_elements = self._find_elements_in_parent(element, self.selectors['tweet_link'])
            if link_elements:
                href = link_elements[0].get_attribute('href')
                if href:
                    tweet_data['url'] = href
            
            # Store raw HTML for debugging
            try:
                tweet_data['raw_html'] = element.get_attribute('outerHTML')[:500]
            except:
                pass
            
            return tweet_data
            
        except Exception as e:
            logging.debug(f"[TwitterEngine] Error extracting tweet data: {e}")
            return None
    
    def _extract_timestamp(self, element) -> str:
        """
        Extract timestamp from tweet element
        
        Args:
            element: WebElement representing the tweet
            
        Returns:
            ISO format timestamp string
        """
        try:
            # Try to find time element
            time_elements = self._find_elements_in_parent(element, self.selectors['tweet_timestamp'])
            
            if time_elements:
                time_elem = time_elements[0]
                
                # Try to get datetime attribute
                datetime_attr = time_elem.get_attribute('datetime')
                if datetime_attr:
                    return datetime_attr
                
                # Try to parse text content
                time_text = time_elem.text.strip()
                if time_text:
                    # Parse relative time (e.g., "2h", "3d", "1w")
                    if 's' in time_text or 'second' in time_text:
                        seconds = int(re.search(r'\d+', time_text).group()) if re.search(r'\d+', time_text) else 30
                        timestamp = datetime.now() - timedelta(seconds=seconds)
                        return timestamp.isoformat()
                    elif 'm' in time_text or 'minute' in time_text:
                        minutes = int(re.search(r'\d+', time_text).group()) if re.search(r'\d+', time_text) else 1
                        timestamp = datetime.now() - timedelta(minutes=minutes)
                        return timestamp.isoformat()
                    elif 'h' in time_text or 'hour' in time_text:
                        hours = int(re.search(r'\d+', time_text).group()) if re.search(r'\d+', time_text) else 1
                        timestamp = datetime.now() - timedelta(hours=hours)
                        return timestamp.isoformat()
                    elif 'd' in time_text or 'day' in time_text:
                        days = int(re.search(r'\d+', time_text).group()) if re.search(r'\d+', time_text) else 1
                        timestamp = datetime.now() - timedelta(days=days)
                        return timestamp.isoformat()
                    elif 'w' in time_text or 'week' in time_text:
                        weeks = int(re.search(r'\d+', time_text).group()) if re.search(r'\d+', time_text) else 1
                        timestamp = datetime.now() - timedelta(weeks=weeks)
                        return timestamp.isoformat()
            
            # If no timestamp found, use current time
            return datetime.now().isoformat()
            
        except Exception as e:
            logging.debug(f"[TwitterEngine] Error extracting timestamp: {e}")
            return datetime.now().isoformat()
    
    def _extract_hashtags(self, element) -> List[str]:
        """
        Extract hashtags from tweet element
        
        Args:
            element: WebElement representing the tweet
            
        Returns:
            List of hashtag strings (without # symbol)
        """
        hashtags = []
        
        try:
            # Find hashtag elements
            hashtag_elements = self._find_elements_in_parent(element, self.selectors['hashtag'])
            
            for elem in hashtag_elements:
                try:
                    # Get href attribute
                    href = elem.get_attribute('href')
                    if href and '/hashtag/' in href:
                        # Extract hashtag from URL
                        hashtag = href.split('/hashtag/')[-1].split('?')[0]
                        hashtags.append(hashtag)
                    else:
                        # Try to get text content
                        text = elem.text.strip()
                        if text.startswith('#'):
                            hashtags.append(text[1:])
                except:
                    continue
            
            # Also extract hashtags from text using regex
            text_elements = self._find_elements_in_parent(element, self.selectors['tweet_text'])
            if text_elements:
                full_text = ' '.join([elem.text for elem in text_elements])
                regex_hashtags = re.findall(r'#(\w+)', full_text)
                hashtags.extend(regex_hashtags)
            
            # Remove duplicates and return
            return list(set(hashtags))
            
        except Exception as e:
            logging.debug(f"[TwitterEngine] Error extracting hashtags: {e}")
            return []
    
    def _extract_mentions(self, element) -> List[str]:
        """
        Extract mentions from tweet element
        
        Args:
            element: WebElement representing the tweet
            
        Returns:
            List of mentioned usernames (without @ symbol)
        """
        mentions = []
        
        try:
            # Find mention elements
            mention_elements = self._find_elements_in_parent(element, self.selectors['mention'])
            
            for elem in mention_elements:
                try:
                    # Get href attribute
                    href = elem.get_attribute('href')
                    if href:
                        # Extract username from URL
                        username = href.rstrip('/').split('/')[-1]
                        if username.startswith('@'):
                            username = username[1:]
                        mentions.append(username)
                    else:
                        # Try to get text content
                        text = elem.text.strip()
                        if text.startswith('@'):
                            mentions.append(text[1:])
                except:
                    continue
            
            # Also extract mentions from text using regex
            text_elements = self._find_elements_in_parent(element, self.selectors['tweet_text'])
            if text_elements:
                full_text = ' '.join([elem.text for elem in text_elements])
                regex_mentions = re.findall(r'@(\w+)', full_text)
                mentions.extend(regex_mentions)
            
            # Remove duplicates and return
            return list(set(mentions))
            
        except Exception as e:
            logging.debug(f"[TwitterEngine] Error extracting mentions: {e}")
            return []
    
    def _check_rate_limit(self):
        """
        Check and enforce rate limiting
        Implements exponential backoff when rate limit is approached
        """
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [
                t for t in self.request_times
                if current_time - t < 60
            ]
            
            # Check if we're at the rate limit
            if len(self.request_times) >= self.rate_limit_config['requests_per_minute']:
                # Calculate wait time
                oldest_request = min(self.request_times)
                wait_time = 60 - (current_time - oldest_request)
                
                if wait_time > 0:
                    logging.warning(
                        f"[TwitterEngine] Rate limit reached, waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    
                    # Clear old requests after waiting
                    current_time = time.time()
                    self.request_times = [
                        t for t in self.request_times
                        if current_time - t < 60
                    ]
            
            # Record this request
            self.request_times.append(current_time)
    
    def _handle_rate_limit(self, retry_count: int = 0):
        """
        Handle rate limit errors with exponential backoff
        
        Args:
            retry_count: Current retry attempt number
        """
        self._update_stats('rate_limits_hit')
        
        if retry_count >= self.rate_limit_config['max_retries']:
            logging.error(
                f"[TwitterEngine] Max retries ({self.rate_limit_config['max_retries']}) "
                "reached for rate limit"
            )
            return
        
        # Calculate backoff time (exponential)
        backoff_time = min(
            self.rate_limit_config['backoff_base'] * (2 ** retry_count),
            self.rate_limit_config['max_backoff']
        )
        
        logging.warning(
            f"[TwitterEngine] Rate limit hit, backing off for {backoff_time}s "
            f"(attempt {retry_count + 1}/{self.rate_limit_config['max_retries']})"
        )
        
        time.sleep(backoff_time)
    
    def _find_elements(self, driver, selectors: List[str]):
        """
        Try multiple selectors to find elements
        
        Args:
            driver: WebDriver instance
            selectors: List of CSS selectors to try
            
        Returns:
            List of found WebElements
        """
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    return elements
            except:
                continue
        
        return []
    
    def _find_elements_in_parent(self, parent_element, selectors: List[str]):
        """
        Try multiple selectors to find elements within a parent element
        
        Args:
            parent_element: Parent WebElement
            selectors: List of CSS selectors to try
            
        Returns:
            List of found WebElements
        """
        for selector in selectors:
            try:
                elements = parent_element.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    return elements
            except:
                continue
        
        return []
    
    def _is_crime_related(self, text: str) -> bool:
        """
        Check if text is crime-related and mentions Karachi
        
        Args:
            text: Text to check
            
        Returns:
            True if text is crime-related and mentions Karachi
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Must mention Karachi or a Karachi area
        has_karachi = any(area.lower() in text_lower for area in self.karachi_areas)
        
        if not has_karachi:
            return False
        
        # Must have crime keywords
        has_crime = any(keyword.lower() in text_lower for keyword in self.crime_keywords)
        
        return has_crime
    
    def _update_stats(self, key: str, increment: int = 1):
        """Update statistics"""
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + increment
    
    def get_stats(self) -> Dict:
        """
        Get scraping statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'total_tweets_found': 0,
                'valid_tweets_extracted': 0,
                'threads_extracted': 0,
                'hashtags_tracked': 0,
                'mentions_tracked': 0,
                'rate_limits_hit': 0,
                'searches_performed': 0
            }
            logging.info("[TwitterEngine] Statistics reset")
    
    def scrape_all_queries(self, max_tweets_per_query: int = 50) -> List[Dict]:
        """
        Scrape all predefined search queries
        
        Args:
            max_tweets_per_query: Maximum tweets to collect per query
            
        Returns:
            List of all extracted tweets
        """
        all_tweets = []
        
        for query in self.search_queries:
            try:
                tweets = self.scrape_search(query, max_tweets=max_tweets_per_query)
                all_tweets.extend(tweets)
                
                # Add delay between queries to avoid rate limiting
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"[TwitterEngine] Error scraping query '{query}': {e}")
                continue
        
        logging.info(
            f"[TwitterEngine] Completed all queries: {len(all_tweets)} total tweets"
        )
        
        return all_tweets


# ============================================================================
# TASK 12: REDDIT SCRAPING ENGINE USING PRAW
# ============================================================================

class RedditEngine:
    """
    Reddit scraping engine using PRAW (Python Reddit API Wrapper) for crime incident collection.
    
    Features:
    - PRAW-based API integration for reliable Reddit access
    - Subreddit monitoring (r/karachi and r/pakistan)
    - Keyword and location filtering for crime-related posts
    - Post and comment extraction for comprehensive incident details
    - Confidence scoring based on upvotes and engagement
    - Rate limit handling (60 requests/minute)
    - Area-specific search functionality
    - Integration with error recovery system
    
    Requirements addressed: 31, 32, 33
    """
    
    def __init__(self, config: Configuration, error_recovery: Optional[ErrorRecoverySystem] = None):
        """
        Initialize Reddit scraping engine with PRAW
        
        Args:
            config: Configuration instance with Reddit API credentials
            error_recovery: Optional ErrorRecoverySystem for error handling
        """
        self.config = config
        self.error_recovery = error_recovery
        self.reddit = None
        
        # Subreddits to monitor
        self.subreddits = ['karachi', 'pakistan']
        
        # Crime-related keywords for filtering
        self.crime_keywords = [
            'snatch', 'snatching', 'snatched',
            'theft', 'stolen', 'stole', 'steal',
            'robbery', 'robbed', 'rob',
            'mobile', 'phone', 'cell', 'smartphone',
            'crime', 'incident',
            'mugging', 'mugged',
            'dacoity', 'loot', 'looted',
            'purse', 'wallet', 'bag',
            'bike', 'motorcycle', 'car',
            'armed', 'weapon', 'gun', 'knife'
        ]
        
        # Karachi area keywords for location filtering
        self.karachi_areas = [
            'karachi', 'dha', 'clifton', 'saddar', 'gulshan', 'nazimabad',
            'north nazimabad', 'malir', 'korangi', 'landhi',
            'orangi', 'liaquatabad', 'shah faisal', 'bin qasim',
            'kemari', 'lyari', 'jamshed', 'garden', 'baldia',
            'site', 'north karachi', 'new karachi', 'federal b area',
            'pechs', 'bahadurabad', 'tariq road', 'shahrah-e-faisal',
            'defence', 'phase', 'gulistan-e-johar', 'malir cantt',
            'scheme 33', 'buffer zone', 'surjani', 'north nazimabad',
            'fb area', 'kda', 'soldier bazaar', 'burns road',
            'empress market', 'tower', 'kharadar', 'mithadar'
        ]
        
        # Rate limiting configuration (Reddit API: 60 requests/minute)
        self.rate_limit_config = {
            'requests_per_minute': 60,
            'backoff_base': 60,  # seconds
            'max_backoff': 300,  # 5 minutes
            'max_retries': 3
        }
        
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'total_posts_found': 0,
            'valid_posts_extracted': 0,
            'comments_extracted': 0,
            'rate_limits_hit': 0,
            'searches_performed': 0,
            'subreddits_scraped': 0
        }
        self.stats_lock = threading.Lock()
        
        # Initialize PRAW
        self._initialize_praw()
        
        logging.info("[RedditEngine] Initialized with PRAW API")
    
    def _initialize_praw(self):
        """Initialize PRAW Reddit API client"""
        if not HAS_PRAW:
            logging.error("[RedditEngine] PRAW not installed - Reddit scraping disabled")
            return
        
        if not self.config.has_reddit_api():
            logging.warning("[RedditEngine] Reddit API credentials not configured - Reddit scraping disabled")
            return
        
        try:
            self.reddit = praw.Reddit(
                client_id=self.config.REDDIT_CLIENT_ID,
                client_secret=self.config.REDDIT_CLIENT_SECRET,
                user_agent=self.config.REDDIT_USER_AGENT
            )
            
            # Test connection
            _ = self.reddit.user.me()
            logging.info("[RedditEngine] PRAW initialized successfully")
            
        except Exception as e:
            logging.error(f"[RedditEngine] Failed to initialize PRAW: {e}")
            self.reddit = None
    
    def scrape_subreddit(
        self,
        subreddit_name: str,
        time_filter: str = 'week',
        limit: int = 100
    ) -> List[Dict]:
        """
        Scrape posts from a subreddit with crime filtering
        
        Args:
            subreddit_name: Name of subreddit (e.g., 'karachi')
            time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
            limit: Maximum number of posts to retrieve (default: 100)
            
        Returns:
            List of extracted post dictionaries
        """
        if not self.reddit:
            logging.warning("[RedditEngine] Reddit API not initialized, skipping scrape")
            return []
        
        logging.info(f"[RedditEngine] Scraping r/{subreddit_name} (time_filter={time_filter}, limit={limit})")
        
        posts = []
        
        try:
            # Check rate limiting
            self._check_rate_limit()
            
            # Get subreddit
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get new posts from the time period
            submissions = subreddit.new(limit=limit)
            
            # Filter and extract posts
            for submission in submissions:
                try:
                    # Check rate limiting
                    self._check_rate_limit()
                    
                    self._update_stats('total_posts_found')
                    
                    # Filter for crime-related Karachi content
                    if self._is_crime_related_karachi(submission):
                        # Extract post data
                        post_data = self.extract_post_data(submission)
                        
                        if post_data:
                            # Extract comments for additional details
                            comments = self.extract_comments(submission)
                            post_data['comments'] = comments
                            post_data['comment_count'] = len(comments)
                            
                            # Calculate confidence score
                            post_data['confidence'] = self.calculate_confidence(submission)
                            
                            posts.append(post_data)
                            self._update_stats('valid_posts_extracted')
                            
                            logging.info(
                                f"[RedditEngine] Extracted valid post: {submission.id} "
                                f"(confidence: {post_data['confidence']:.2f})"
                            )
                
                except Exception as e:
                    logging.debug(f"[RedditEngine] Error processing submission: {e}")
                    continue
            
            self._update_stats('subreddits_scraped')
            logging.info(
                f"[RedditEngine] Completed r/{subreddit_name}: {len(posts)} valid posts"
            )
            
        except Exception as e:
            logging.error(f"[RedditEngine] Error scraping r/{subreddit_name}: {e}")
            logging.error(traceback.format_exc())
            
            # Check if it's a rate limit error
            if 'rate' in str(e).lower() or '429' in str(e):
                self.handle_rate_limit()
        
        return posts
    
    def filter_crime_posts(self, submissions: List) -> List:
        """
        Filter submissions for Karachi crime-related content
        
        Args:
            submissions: List of PRAW submission objects
            
        Returns:
            List of filtered submissions
        """
        filtered = []
        
        for submission in submissions:
            if self._is_crime_related_karachi(submission):
                filtered.append(submission)
        
        logging.info(
            f"[RedditEngine] Filtered {len(filtered)}/{len(submissions)} crime-related posts"
        )
        
        return filtered
    
    def extract_post_data(self, submission) -> Optional[Dict]:
        """
        Extract data from Reddit post including title, selftext, and metadata
        
        Args:
            submission: PRAW submission object
            
        Returns:
            Dictionary with extracted post data or None if extraction failed
        """
        try:
            # Extract basic post data
            post_data = {
                'post_id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'author': str(submission.author) if submission.author else '[deleted]',
                'subreddit': str(submission.subreddit),
                'url': f"https://reddit.com{submission.permalink}",
                'created_utc': submission.created_utc,
                'timestamp': datetime.fromtimestamp(submission.created_utc).isoformat(),
                'upvotes': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'is_self': submission.is_self,
                'link_url': submission.url if not submission.is_self else None,
                'source': 'Reddit',
                'raw_text': f"{submission.title}\n\n{submission.selftext}"
            }
            
            # Extract flair if available
            if submission.link_flair_text:
                post_data['flair'] = submission.link_flair_text
            
            return post_data
            
        except Exception as e:
            logging.debug(f"[RedditEngine] Error extracting post data: {e}")
            return None
    
    def extract_comments(self, submission, max_comments: int = 20) -> List[Dict]:
        """
        Extract top-level comments for additional incident details
        
        Args:
            submission: PRAW submission object
            max_comments: Maximum number of comments to extract (default: 20)
            
        Returns:
            List of comment dictionaries with relevant details
        """
        comments = []
        
        try:
            # Check rate limiting
            self._check_rate_limit()
            
            # Replace "MoreComments" objects to get all comments
            submission.comments.replace_more(limit=0)
            
            # Extract top-level comments
            for comment in submission.comments[:max_comments]:
                try:
                    # Skip deleted/removed comments
                    if not comment.body or comment.body in ['[deleted]', '[removed]']:
                        continue
                    
                    # Check if comment contains relevant information
                    if self._is_relevant_comment(comment.body):
                        comment_data = {
                            'comment_id': comment.id,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'body': comment.body,
                            'created_utc': comment.created_utc,
                            'timestamp': datetime.fromtimestamp(comment.created_utc).isoformat(),
                            'score': comment.score,
                            'is_submitter': comment.is_submitter
                        }
                        
                        comments.append(comment_data)
                        self._update_stats('comments_extracted')
                
                except Exception as e:
                    logging.debug(f"[RedditEngine] Error extracting comment: {e}")
                    continue
            
            logging.debug(
                f"[RedditEngine] Extracted {len(comments)} relevant comments from post {submission.id}"
            )
            
        except Exception as e:
            logging.debug(f"[RedditEngine] Error extracting comments: {e}")
        
        return comments
    
    def calculate_confidence(self, submission) -> float:
        """
        Calculate confidence score based on upvotes and engagement
        
        Scoring:
        - Base score: 0.5 (community-reported)
        - +0.1 for every 10 upvotes (max +0.3)
        - +0.1 if OP provides details (long post with specifics)
        - +0.1 if post has high engagement (comments)
        
        Args:
            submission: PRAW submission object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base score for community-reported
        
        try:
            # Upvote bonus (max +0.3)
            upvote_bonus = min(0.3, (submission.score / 10) * 0.1)
            confidence += upvote_bonus
            
            # Detail bonus (+0.1 if post has substantial content)
            text_length = len(submission.title) + len(submission.selftext)
            if text_length > 200:  # Substantial post
                confidence += 0.1
            
            # Engagement bonus (+0.1 if post has good engagement)
            if submission.num_comments >= 5:
                confidence += 0.1
            
            # Cap at 1.0
            confidence = min(1.0, confidence)
            
        except Exception as e:
            logging.debug(f"[RedditEngine] Error calculating confidence: {e}")
        
        return confidence
    
    def handle_rate_limit(self, retry_count: int = 0):
        """
        Handle Reddit API rate limiting with exponential backoff
        
        Args:
            retry_count: Current retry attempt number
        """
        self._update_stats('rate_limits_hit')
        
        if retry_count >= self.rate_limit_config['max_retries']:
            logging.error(
                f"[RedditEngine] Max retries ({self.rate_limit_config['max_retries']}) "
                "reached for rate limit"
            )
            return
        
        # Calculate backoff time (exponential)
        backoff_time = min(
            self.rate_limit_config['backoff_base'] * (2 ** retry_count),
            self.rate_limit_config['max_backoff']
        )
        
        logging.warning(
            f"[RedditEngine] Rate limit hit, backing off for {backoff_time}s "
            f"(attempt {retry_count + 1}/{self.rate_limit_config['max_retries']})"
        )
        
        time.sleep(backoff_time)
    
    def search_area_specific(self, area: str, time_filter: str = 'week', limit: int = 50) -> List[Dict]:
        """
        Search for area-specific crime posts across monitored subreddits
        
        Args:
            area: Karachi area to search for (e.g., 'DHA', 'Clifton')
            time_filter: Time filter for search
            limit: Maximum results per subreddit
            
        Returns:
            List of extracted post dictionaries
        """
        logging.info(f"[RedditEngine] Searching for area-specific posts: {area}")
        
        all_posts = []
        
        # Construct search query
        search_query = f"Karachi {area} (snatching OR theft OR robbery OR crime OR stolen)"
        
        for subreddit_name in self.subreddits:
            try:
                if not self.reddit:
                    continue
                
                # Check rate limiting
                self._check_rate_limit()
                
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search subreddit
                results = subreddit.search(
                    search_query,
                    time_filter=time_filter,
                    limit=limit
                )
                
                # Process results
                for submission in results:
                    try:
                        # Check rate limiting
                        self._check_rate_limit()
                        
                        # Filter for crime-related content
                        if self._is_crime_related_karachi(submission):
                            post_data = self.extract_post_data(submission)
                            
                            if post_data:
                                # Extract comments
                                comments = self.extract_comments(submission)
                                post_data['comments'] = comments
                                post_data['comment_count'] = len(comments)
                                
                                # Calculate confidence
                                post_data['confidence'] = self.calculate_confidence(submission)
                                post_data['search_area'] = area
                                
                                all_posts.append(post_data)
                                self._update_stats('valid_posts_extracted')
                    
                    except Exception as e:
                        logging.debug(f"[RedditEngine] Error processing search result: {e}")
                        continue
                
                self._update_stats('searches_performed')
                
            except Exception as e:
                logging.error(f"[RedditEngine] Error searching r/{subreddit_name} for {area}: {e}")
                continue
        
        logging.info(
            f"[RedditEngine] Area search completed for '{area}': {len(all_posts)} posts"
        )
        
        return all_posts
    
    def scrape_all_subreddits(self, time_filter: str = 'week', limit: int = 100) -> List[Dict]:
        """
        Scrape all monitored subreddits
        
        Args:
            time_filter: Time filter for posts
            limit: Maximum posts per subreddit
            
        Returns:
            List of all extracted posts
        """
        all_posts = []
        
        for subreddit_name in self.subreddits:
            try:
                posts = self.scrape_subreddit(subreddit_name, time_filter, limit)
                all_posts.extend(posts)
                
                # Add delay between subreddits to be polite
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"[RedditEngine] Error scraping r/{subreddit_name}: {e}")
                continue
        
        logging.info(
            f"[RedditEngine] Completed all subreddits: {len(all_posts)} total posts"
        )
        
        return all_posts
    
    def _is_crime_related_karachi(self, submission) -> bool:
        """
        Check if submission is crime-related and mentions Karachi
        
        Args:
            submission: PRAW submission object
            
        Returns:
            True if submission is relevant
        """
        try:
            # Combine title and selftext for checking
            text = f"{submission.title} {submission.selftext}".lower()
            
            # Must mention Karachi or a Karachi area
            has_karachi = any(area.lower() in text for area in self.karachi_areas)
            
            if not has_karachi:
                return False
            
            # Must have crime keywords
            has_crime = any(keyword.lower() in text for keyword in self.crime_keywords)
            
            # Additional check: minimum upvotes for credibility (at least 2)
            has_credibility = submission.score >= 2
            
            return has_crime and has_credibility
            
        except Exception as e:
            logging.debug(f"[RedditEngine] Error checking relevance: {e}")
            return False
    
    def _is_relevant_comment(self, comment_text: str) -> bool:
        """
        Check if comment contains relevant incident details
        
        Args:
            comment_text: Comment body text
            
        Returns:
            True if comment is relevant
        """
        if not comment_text or len(comment_text) < 20:
            return False
        
        text_lower = comment_text.lower()
        
        # Check for location mentions
        has_location = any(area.lower() in text_lower for area in self.karachi_areas)
        
        # Check for incident details (time, device, description)
        has_details = any(keyword in text_lower for keyword in [
            'time', 'date', 'yesterday', 'today', 'last night',
            'phone', 'mobile', 'device', 'model',
            'happened', 'incident', 'stolen', 'snatched'
        ])
        
        return has_location or has_details
    
    def _check_rate_limit(self):
        """
        Check and enforce rate limiting (60 requests/minute for Reddit API)
        """
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [
                t for t in self.request_times
                if current_time - t < 60
            ]
            
            # Check if we're at the rate limit
            if len(self.request_times) >= self.rate_limit_config['requests_per_minute']:
                # Calculate wait time
                oldest_request = min(self.request_times)
                wait_time = 60 - (current_time - oldest_request)
                
                if wait_time > 0:
                    logging.warning(
                        f"[RedditEngine] Rate limit reached, waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    
                    # Clear old requests after waiting
                    current_time = time.time()
                    self.request_times = [
                        t for t in self.request_times
                        if current_time - t < 60
                    ]
            
            # Record this request
            self.request_times.append(current_time)
    
    def _update_stats(self, key: str, increment: int = 1):
        """Update statistics"""
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + increment
    
    def get_stats(self) -> Dict:
        """
        Get scraping statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'total_posts_found': 0,
                'valid_posts_extracted': 0,
                'comments_extracted': 0,
                'rate_limits_hit': 0,
                'searches_performed': 0,
                'subreddits_scraped': 0
            }
            logging.info("[RedditEngine] Statistics reset")


# ============================================================================
# TASK 13: RSS FEED ENGINE WITH DISCOVERY
# ============================================================================

class RSSEngine:
    """
    RSS feed engine with automatic discovery and monitoring.
    
    Features:
    - feedparser integration for reliable RSS parsing
    - Automatic RSS feed discovery on websites
    - Feed validation for relevant content
    - Feed health tracking and frequency adjustment
    - Entry deduplication mechanism
    - Integration with error recovery system
    
    Requirements addressed: 10, 27
    """
    
    def __init__(self, config: Configuration, error_recovery: Optional[ErrorRecoverySystem] = None):
        """
        Initialize RSS feed engine
        
        Args:
            config: Configuration instance
            error_recovery: Optional ErrorRecoverySystem for error handling
        """
        self.config = config
        self.error_recovery = error_recovery
        
        # Known RSS feeds for Pakistani news sites
        self.feeds = {
            'dawn_karachi': 'https://www.dawn.com/feeds/karachi',
            'dawn_pakistan': 'https://www.dawn.com/feeds/pakistan',
            'geo_news': 'https://www.geo.tv/rss/1/1',
            'tribune_karachi': 'https://tribune.com.pk/rss/karachi',
            'express_karachi': 'https://www.express.pk/rss/karachi',
        }
        
        # Feed health tracking
        self.feed_health = {}
        self._initialize_feed_health()
        
        # Entry deduplication cache (URL-based)
        self.seen_entries = set()
        
        # Crime-related keywords for filtering
        self.crime_keywords = [
            'snatch', 'snatching', 'snatched',
            'theft', 'stolen', 'stole', 'steal',
            'robbery', 'robbed', 'rob',
            'mobile', 'phone', 'cell', 'smartphone',
            'crime', 'incident',
            'mugging', 'mugged',
            'dacoity', 'loot', 'looted',
            'purse', 'wallet', 'bag',
            'bike', 'motorcycle', 'car',
            'armed', 'weapon', 'gun', 'knife',
            'karachi'
        ]
        
        # Statistics tracking
        self.stats = {
            'feeds_discovered': 0,
            'feeds_validated': 0,
            'feeds_scraped': 0,
            'entries_found': 0,
            'entries_extracted': 0,
            'duplicates_skipped': 0
        }
        self.stats_lock = threading.Lock()
        
        logging.info("[RSSEngine] Initialized with feedparser")
    
    def _initialize_feed_health(self):
        """Initialize health tracking for all known feeds"""
        for feed_name, feed_url in self.feeds.items():
            self.feed_health[feed_url] = {
                'name': feed_name,
                'url': feed_url,
                'active': True,
                'success_count': 0,
                'failure_count': 0,
                'last_success': None,
                'last_failure': None,
                'consecutive_failures': 0,
                'total_entries': 0,
                'valid_entries': 0,
                'check_frequency': 1,  # Check every run initially
                'last_checked': None
            }
    
    def discover_feeds(self, website_url: str) -> List[str]:
        """
        Discover RSS feed links on a website
        
        Searches for RSS/Atom feed links in:
        - <link> tags in HTML head
        - Common RSS URL patterns
        - Links with RSS-related text
        
        Args:
            website_url: URL of website to search for feeds
            
        Returns:
            List of discovered feed URLs
        """
        logging.info(f"[RSSEngine] Discovering feeds on: {website_url}")
        
        discovered_feeds = []
        
        try:
            # Make HTTP request with timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(website_url, headers=headers, timeout=15, verify=False)
            response.raise_for_status()
            
            html_content = response.text
            
            # Method 1: Look for <link> tags with RSS/Atom type
            link_patterns = [
                r'<link[^>]*type=["\']application/rss\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                r'<link[^>]*href=["\']([^"\']+)["\'][^>]*type=["\']application/rss\+xml["\']',
                r'<link[^>]*type=["\']application/atom\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                r'<link[^>]*href=["\']([^"\']+)["\'][^>]*type=["\']application/atom\+xml["\']',
            ]
            
            for pattern in link_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for match in matches:
                    # Convert relative URLs to absolute
                    if match.startswith('http'):
                        feed_url = match
                    elif match.startswith('/'):
                        from urllib.parse import urljoin
                        feed_url = urljoin(website_url, match)
                    else:
                        feed_url = f"{website_url.rstrip('/')}/{match}"
                    
                    if feed_url not in discovered_feeds:
                        discovered_feeds.append(feed_url)
            
            # Method 2: Look for common RSS URL patterns
            common_patterns = [
                '/rss', '/feed', '/feeds', '/rss.xml', '/feed.xml',
                '/atom.xml', '/index.xml', '/rss/news', '/feeds/posts'
            ]
            
            base_url = website_url.rstrip('/')
            for pattern in common_patterns:
                potential_feed = f"{base_url}{pattern}"
                
                # Quick check if this URL exists
                try:
                    test_response = requests.head(potential_feed, headers=headers, timeout=5, verify=False)
                    if test_response.status_code == 200:
                        if potential_feed not in discovered_feeds:
                            discovered_feeds.append(potential_feed)
                except:
                    pass
            
            # Method 3: Look for links with RSS-related text
            rss_link_pattern = r'<a[^>]*href=["\']([^"\']*(?:rss|feed|atom)[^"\']*)["\']'
            matches = re.findall(rss_link_pattern, html_content, re.IGNORECASE)
            
            for match in matches:
                if match.startswith('http'):
                    feed_url = match
                elif match.startswith('/'):
                    from urllib.parse import urljoin
                    feed_url = urljoin(website_url, match)
                else:
                    continue
                
                if feed_url not in discovered_feeds:
                    discovered_feeds.append(feed_url)
            
            self._update_stats('feeds_discovered', len(discovered_feeds))
            
            logging.info(
                f"[RSSEngine] Discovered {len(discovered_feeds)} potential feeds on {website_url}"
            )
            
        except Exception as e:
            logging.error(f"[RSSEngine] Error discovering feeds on {website_url}: {e}")
        
        return discovered_feeds
    
    def scrape_feed(self, feed_url: str, max_entries: int = 50) -> List[Dict]:
        """
        Parse RSS feed and extract entries
        
        Args:
            feed_url: URL of RSS/Atom feed
            max_entries: Maximum number of entries to extract (default: 50)
            
        Returns:
            List of extracted entry dictionaries
        """
        logging.info(f"[RSSEngine] Scraping feed: {feed_url}")
        
        entries = []
        
        try:
            # Parse feed with feedparser
            feed = feedparser.parse(feed_url)
            
            # Check if feed was parsed successfully
            if feed.bozo and not feed.entries:
                # Feed has errors and no entries
                logging.warning(f"[RSSEngine] Failed to parse feed {feed_url}: {feed.bozo_exception}")
                self._record_feed_failure(feed_url)
                return []
            
            # Update feed health
            self._record_feed_success(feed_url)
            
            # Extract feed metadata
            feed_title = feed.feed.get('title', 'Unknown Feed')
            feed_link = feed.feed.get('link', feed_url)
            
            logging.info(f"[RSSEngine] Feed: {feed_title} ({len(feed.entries)} entries)")
            
            # Process entries
            for entry in feed.entries[:max_entries]:
                try:
                    self._update_stats('entries_found')
                    
                    # Extract entry data
                    entry_data = self._extract_entry_data(entry, feed_url, feed_title)
                    
                    if not entry_data:
                        continue
                    
                    # Check for duplicates
                    entry_id = entry_data.get('entry_url', entry_data.get('entry_id', ''))
                    
                    if entry_id in self.seen_entries:
                        self._update_stats('duplicates_skipped')
                        logging.debug(f"[RSSEngine] Skipping duplicate entry: {entry_id}")
                        continue
                    
                    # Filter for relevant content (Karachi crime)
                    if self._is_relevant_entry(entry_data):
                        entries.append(entry_data)
                        self.seen_entries.add(entry_id)
                        self._update_stats('entries_extracted')
                        
                        # Update feed health
                        if feed_url in self.feed_health:
                            self.feed_health[feed_url]['valid_entries'] += 1
                        
                        logging.info(
                            f"[RSSEngine] Extracted entry: {entry_data.get('title', 'No title')[:60]}"
                        )
                
                except Exception as e:
                    logging.debug(f"[RSSEngine] Error processing entry: {e}")
                    continue
            
            # Update feed health
            if feed_url in self.feed_health:
                self.feed_health[feed_url]['total_entries'] += len(feed.entries)
                self.feed_health[feed_url]['last_checked'] = datetime.now().isoformat()
            
            self._update_stats('feeds_scraped')
            
            logging.info(
                f"[RSSEngine] Completed feed {feed_url}: {len(entries)} relevant entries"
            )
            
        except Exception as e:
            logging.error(f"[RSSEngine] Error scraping feed {feed_url}: {e}")
            logging.error(traceback.format_exc())
            self._record_feed_failure(feed_url)
        
        return entries
    
    def validate_feed(self, feed_url: str) -> bool:
        """
        Validate that feed contains relevant Karachi crime content
        
        Args:
            feed_url: URL of feed to validate
            
        Returns:
            True if feed is valid and relevant, False otherwise
        """
        logging.info(f"[RSSEngine] Validating feed: {feed_url}")
        
        try:
            # Parse feed
            feed = feedparser.parse(feed_url)
            
            # Check if feed parsed successfully
            if feed.bozo and not feed.entries:
                logging.warning(f"[RSSEngine] Feed validation failed - cannot parse: {feed_url}")
                return False
            
            # Check if feed has entries
            if not feed.entries:
                logging.warning(f"[RSSEngine] Feed validation failed - no entries: {feed_url}")
                return False
            
            # Check if any entries are relevant
            relevant_count = 0
            
            for entry in feed.entries[:20]:  # Check first 20 entries
                entry_data = self._extract_entry_data(entry, feed_url, feed.feed.get('title', ''))
                
                if entry_data and self._is_relevant_entry(entry_data):
                    relevant_count += 1
            
            # Feed is valid if at least 10% of entries are relevant
            is_valid = relevant_count >= max(1, len(feed.entries[:20]) * 0.1)
            
            if is_valid:
                self._update_stats('feeds_validated')
                logging.info(
                    f"[RSSEngine] Feed validated: {feed_url} "
                    f"({relevant_count}/{len(feed.entries[:20])} relevant)"
                )
            else:
                logging.warning(
                    f"[RSSEngine] Feed validation failed - insufficient relevant content: {feed_url} "
                    f"({relevant_count}/{len(feed.entries[:20])} relevant)"
                )
            
            return is_valid
            
        except Exception as e:
            logging.error(f"[RSSEngine] Error validating feed {feed_url}: {e}")
            return False
    
    def scrape_all_feeds(self, max_entries_per_feed: int = 50) -> List[Dict]:
        """
        Scrape all active feeds with frequency adjustment
        
        Args:
            max_entries_per_feed: Maximum entries to extract per feed
            
        Returns:
            List of all extracted entries
        """
        logging.info("[RSSEngine] Scraping all active feeds")
        
        all_entries = []
        
        for feed_url, health in self.feed_health.items():
            # Skip inactive feeds
            if not health['active']:
                logging.debug(f"[RSSEngine] Skipping inactive feed: {feed_url}")
                continue
            
            # Check if feed should be scraped based on frequency
            if not self._should_scrape_feed(feed_url):
                logging.debug(
                    f"[RSSEngine] Skipping feed (frequency): {feed_url} "
                    f"(check_frequency={health['check_frequency']})"
                )
                continue
            
            try:
                # Scrape feed
                entries = self.scrape_feed(feed_url, max_entries_per_feed)
                all_entries.extend(entries)
                
                # Adjust frequency based on results
                self._adjust_feed_frequency(feed_url, len(entries))
                
                # Add delay between feeds to be polite
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"[RSSEngine] Error scraping feed {feed_url}: {e}")
                continue
        
        logging.info(
            f"[RSSEngine] Completed all feeds: {len(all_entries)} total entries"
        )
        
        return all_entries
    
    def add_feed(self, feed_url: str, feed_name: str = None, validate: bool = True) -> bool:
        """
        Add a new feed to monitoring
        
        Args:
            feed_url: URL of feed to add
            feed_name: Optional name for the feed
            validate: Whether to validate feed before adding (default: True)
            
        Returns:
            True if feed was added successfully
        """
        logging.info(f"[RSSEngine] Adding new feed: {feed_url}")
        
        # Check if feed already exists
        if feed_url in self.feed_health:
            logging.warning(f"[RSSEngine] Feed already exists: {feed_url}")
            return False
        
        # Validate feed if requested
        if validate:
            if not self.validate_feed(feed_url):
                logging.warning(f"[RSSEngine] Feed validation failed, not adding: {feed_url}")
                return False
        
        # Add feed to tracking
        self.feed_health[feed_url] = {
            'name': feed_name or f"feed_{len(self.feed_health)}",
            'url': feed_url,
            'active': True,
            'success_count': 0,
            'failure_count': 0,
            'last_success': None,
            'last_failure': None,
            'consecutive_failures': 0,
            'total_entries': 0,
            'valid_entries': 0,
            'check_frequency': 1,  # Check every run initially
            'last_checked': None
        }
        
        # Add to feeds dictionary
        if feed_name:
            self.feeds[feed_name] = feed_url
        
        logging.info(f"[RSSEngine] Successfully added feed: {feed_url}")
        return True
    
    def _extract_entry_data(self, entry, feed_url: str, feed_title: str) -> Optional[Dict]:
        """
        Extract data from RSS entry
        
        Args:
            entry: feedparser entry object
            feed_url: URL of the feed
            feed_title: Title of the feed
            
        Returns:
            Dictionary with extracted entry data or None
        """
        try:
            # Extract basic entry data
            entry_data = {
                'entry_id': entry.get('id', entry.get('link', '')),
                'entry_url': entry.get('link', ''),
                'title': entry.get('title', ''),
                'summary': entry.get('summary', entry.get('description', '')),
                'content': '',
                'published': entry.get('published', entry.get('updated', '')),
                'author': entry.get('author', ''),
                'feed_url': feed_url,
                'feed_title': feed_title,
                'source': 'RSS',
                'raw_text': ''
            }
            
            # Extract full content if available
            if 'content' in entry and entry.content:
                # feedparser stores content as a list of dicts
                content_parts = [c.get('value', '') for c in entry.content]
                entry_data['content'] = ' '.join(content_parts)
            
            # Build raw text for processing
            entry_data['raw_text'] = f"{entry_data['title']}\n\n{entry_data['summary']}\n\n{entry_data['content']}"
            
            # Parse published date
            if 'published_parsed' in entry and entry.published_parsed:
                try:
                    timestamp = time.mktime(entry.published_parsed)
                    entry_data['timestamp'] = datetime.fromtimestamp(timestamp).isoformat()
                except:
                    entry_data['timestamp'] = datetime.now().isoformat()
            else:
                entry_data['timestamp'] = datetime.now().isoformat()
            
            return entry_data
            
        except Exception as e:
            logging.debug(f"[RSSEngine] Error extracting entry data: {e}")
            return None
    
    def _is_relevant_entry(self, entry_data: Dict) -> bool:
        """
        Check if entry is relevant (Karachi crime-related)
        
        Args:
            entry_data: Entry data dictionary
            
        Returns:
            True if entry is relevant
        """
        # Combine all text fields
        text = f"{entry_data.get('title', '')} {entry_data.get('summary', '')} {entry_data.get('content', '')}".lower()
        
        if not text or len(text) < 20:
            return False
        
        # Must mention Karachi
        has_karachi = 'karachi' in text
        
        if not has_karachi:
            return False
        
        # Must have crime keywords
        has_crime = any(keyword.lower() in text for keyword in self.crime_keywords)
        
        return has_crime
    
    def _should_scrape_feed(self, feed_url: str) -> bool:
        """
        Determine if feed should be scraped based on frequency settings
        
        Args:
            feed_url: URL of feed to check
            
        Returns:
            True if feed should be scraped
        """
        if feed_url not in self.feed_health:
            return True
        
        health = self.feed_health[feed_url]
        
        # Always scrape if never checked
        if health['last_checked'] is None:
            return True
        
        # Check frequency (1 = every run, 2 = every 2 runs, 3 = every 3 runs, etc.)
        # For simplicity, we'll use a counter-based approach
        # In a real implementation, you'd track run numbers
        
        # For now, always scrape if frequency is 1
        if health['check_frequency'] <= 1:
            return True
        
        # Otherwise, use a probabilistic approach based on frequency
        # Lower frequency = lower probability
        probability = 1.0 / health['check_frequency']
        return random.random() < probability
    
    def _adjust_feed_frequency(self, feed_url: str, entries_found: int):
        """
        Adjust feed scraping frequency based on results
        
        If feed consistently has no new entries, reduce check frequency.
        If feed has new entries, increase check frequency.
        
        Args:
            feed_url: URL of feed
            entries_found: Number of relevant entries found in last scrape
        """
        if feed_url not in self.feed_health:
            return
        
        health = self.feed_health[feed_url]
        
        if entries_found > 0:
            # Feed has new content, increase frequency (check more often)
            health['check_frequency'] = max(1, health['check_frequency'] - 0.5)
            logging.debug(
                f"[RSSEngine] Increased frequency for {feed_url}: {health['check_frequency']}"
            )
        else:
            # No new content, decrease frequency (check less often)
            health['check_frequency'] = min(10, health['check_frequency'] + 0.5)
            logging.debug(
                f"[RSSEngine] Decreased frequency for {feed_url}: {health['check_frequency']}"
            )
    
    def _record_feed_success(self, feed_url: str):
        """Record successful feed scrape"""
        if feed_url in self.feed_health:
            health = self.feed_health[feed_url]
            health['success_count'] += 1
            health['last_success'] = datetime.now().isoformat()
            health['consecutive_failures'] = 0
    
    def _record_feed_failure(self, feed_url: str):
        """Record failed feed scrape"""
        if feed_url in self.feed_health:
            health = self.feed_health[feed_url]
            health['failure_count'] += 1
            health['last_failure'] = datetime.now().isoformat()
            health['consecutive_failures'] += 1
            
            # Deactivate feed if too many consecutive failures
            if health['consecutive_failures'] >= 10:
                health['active'] = False
                logging.warning(
                    f"[RSSEngine] Deactivated feed due to consecutive failures: {feed_url}"
                )
    
    def _update_stats(self, key: str, increment: int = 1):
        """Update statistics"""
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + increment
    
    def get_stats(self) -> Dict:
        """
        Get scraping statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return dict(self.stats)
    
    def get_feed_health(self) -> Dict:
        """
        Get health status of all feeds
        
        Returns:
            Dictionary with feed health information
        """
        return dict(self.feed_health)
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'feeds_discovered': 0,
                'feeds_validated': 0,
                'feeds_scraped': 0,
                'entries_found': 0,
                'entries_extracted': 0,
                'duplicates_skipped': 0
            }
            logging.info("[RSSEngine] Statistics reset")


# ============================================================================
# TASK 14: NEWS WEBSITE SCRAPING ENGINE
# ============================================================================

class NewsEngine:
    """
    News website scraping engine for Pakistani news sites with source credibility verification.
    
    Features:
    - Multi-site support (Dawn, Geo, Tribune, Express)
    - Full article content extraction
    - Statistical data parsing and detection
    - Source credibility verification
    - HTTP-based scraping with retry logic
    - Anti-detection measures
    
    Requirements addressed: 18, 20, 27
    """
    
    def __init__(
        self,
        http_client: HTTPClientPool,
        statistical_expander: StatisticalExpander,
        error_recovery: ErrorRecoverySystem
    ):
        """
        Initialize news engine
        
        Args:
            http_client: HTTP client pool for making requests
            statistical_expander: Statistical expander for aggregate reports
            error_recovery: Error recovery system for handling failures
        """
        self.http_client = http_client
        self.statistical_expander = statistical_expander
        self.error_recovery = error_recovery
        
        # Pakistani news sites configuration
        self.news_sites = {
            'dawn': {
                'name': 'Dawn News',
                'base_url': 'https://www.dawn.com',
                'search_url': 'https://www.dawn.com/search?query={query}',
                'article_selectors': [
                    'div.story__content',
                    'article',
                    'div.template__main',
                    'p'
                ],
                'credible': True
            },
            'geo': {
                'name': 'Geo News',
                'base_url': 'https://www.geo.tv',
                'search_url': 'https://www.geo.tv/search/{query}',
                'article_selectors': [
                    'div.content-area',
                    'div.article-content',
                    'article',
                    'p'
                ],
                'credible': True
            },
            'tribune': {
                'name': 'Express Tribune',
                'base_url': 'https://tribune.com.pk',
                'search_url': 'https://tribune.com.pk/search?q={query}',
                'article_selectors': [
                    'div.story-detail',
                    'div.main-content',
                    'article',
                    'p'
                ],
                'credible': True
            },
            'express': {
                'name': 'Express News',
                'base_url': 'https://www.express.com.pk',
                'search_url': 'https://www.express.com.pk/search/?q={query}',
                'article_selectors': [
                    'div.detail-content',
                    'div.story-content',
                    'article',
                    'p'
                ],
                'credible': True
            }
        }
        
        # Crime-related keywords for filtering
        self.crime_keywords = [
            'snatch', 'theft', 'stolen', 'robbery', 'mugging', 'dacoity',
            'mobile', 'phone', 'crime', 'incident', 'loot', 'burglar',
            'street crime', 'armed robbery', 'motorcycle', 'valuables'
        ]
        
        # Karachi location keywords
        self.karachi_keywords = [
            'karachi', 'clifton', 'dha', 'gulshan', 'nazimabad', 'saddar',
            'malir', 'korangi', 'lyari', 'orangi', 'north nazimabad',
            'shah faisal', 'landhi', 'baldia', 'keamari', 'jamshed'
        ]
        
        # Statistics tracking
        self.stats = {
            'dawn_scraped': 0,
            'geo_scraped': 0,
            'tribune_scraped': 0,
            'express_scraped': 0,
            'articles_extracted': 0,
            'statistical_reports_found': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        
        # Seen article URLs to avoid duplicates
        self.seen_urls = set()
        self.seen_urls_lock = threading.Lock()
        
        logging.info("[NewsEngine] Initialized with 4 Pakistani news sites")
    
    def scrape_dawn(self, query: str = "Karachi mobile snatching") -> List[Dict]:
        """
        Scrape Dawn News for crime articles
        
        Args:
            query: Search query (default: "Karachi mobile snatching")
            
        Returns:
            List of article data dictionaries
        """
        logging.info(f"[NewsEngine] Scraping Dawn News for: {query}")
        
        articles = []
        
        try:
            # Build search URL
            search_url = self.news_sites['dawn']['search_url'].format(
                query=query.replace(' ', '+')
            )
            
            # Get search results page
            response = self.http_client.get(search_url, timeout=30)
            
            if not response or response.status_code != 200:
                logging.warning(f"[NewsEngine] Dawn search failed: status {response.status_code if response else 'None'}")
                self._update_stats('errors')
                return articles
            
            # Parse HTML to find article links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links (Dawn uses various link patterns)
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Dawn article URLs typically contain /news/ or /story/
                if '/news/' in href or '/story/' in href:
                    # Make absolute URL
                    if href.startswith('/'):
                        href = self.news_sites['dawn']['base_url'] + href
                    elif not href.startswith('http'):
                        continue
                    
                    # Check if URL is relevant
                    if self._is_relevant_url(href):
                        article_links.append(href)
            
            # Remove duplicates and limit
            article_links = list(set(article_links))[:10]
            
            logging.info(f"[NewsEngine] Found {len(article_links)} Dawn article links")
            
            # Extract content from each article
            for article_url in article_links:
                article_data = self.extract_article(article_url, 'dawn')
                if article_data:
                    articles.append(article_data)
            
            self._update_stats('dawn_scraped')
            
        except Exception as e:
            logging.error(f"[NewsEngine] Error scraping Dawn: {e}")
            self._update_stats('errors')
        
        return articles
    
    def scrape_geo(self, query: str = "Karachi mobile snatching") -> List[Dict]:
        """
        Scrape Geo News for crime articles
        
        Args:
            query: Search query (default: "Karachi mobile snatching")
            
        Returns:
            List of article data dictionaries
        """
        logging.info(f"[NewsEngine] Scraping Geo News for: {query}")
        
        articles = []
        
        try:
            # Build search URL
            search_url = self.news_sites['geo']['search_url'].format(
                query=query.replace(' ', '-')
            )
            
            # Get search results page
            response = self.http_client.get(search_url, timeout=30)
            
            if not response or response.status_code != 200:
                logging.warning(f"[NewsEngine] Geo search failed: status {response.status_code if response else 'None'}")
                self._update_stats('errors')
                return articles
            
            # Parse HTML to find article links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Geo article URLs typically contain /latest/ or /news/
                if '/latest/' in href or '/news/' in href:
                    # Make absolute URL
                    if href.startswith('/'):
                        href = self.news_sites['geo']['base_url'] + href
                    elif not href.startswith('http'):
                        continue
                    
                    # Check if URL is relevant
                    if self._is_relevant_url(href):
                        article_links.append(href)
            
            # Remove duplicates and limit
            article_links = list(set(article_links))[:10]
            
            logging.info(f"[NewsEngine] Found {len(article_links)} Geo article links")
            
            # Extract content from each article
            for article_url in article_links:
                article_data = self.extract_article(article_url, 'geo')
                if article_data:
                    articles.append(article_data)
            
            self._update_stats('geo_scraped')
            
        except Exception as e:
            logging.error(f"[NewsEngine] Error scraping Geo: {e}")
            self._update_stats('errors')
        
        return articles
    
    def scrape_tribune(self, query: str = "Karachi mobile snatching") -> List[Dict]:
        """
        Scrape Express Tribune for crime articles
        
        Args:
            query: Search query (default: "Karachi mobile snatching")
            
        Returns:
            List of article data dictionaries
        """
        logging.info(f"[NewsEngine] Scraping Express Tribune for: {query}")
        
        articles = []
        
        try:
            # Build search URL
            search_url = self.news_sites['tribune']['search_url'].format(
                query=query.replace(' ', '+')
            )
            
            # Get search results page
            response = self.http_client.get(search_url, timeout=30)
            
            if not response or response.status_code != 200:
                logging.warning(f"[NewsEngine] Tribune search failed: status {response.status_code if response else 'None'}")
                self._update_stats('errors')
                return articles
            
            # Parse HTML to find article links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Tribune article URLs typically contain /story/
                if '/story/' in href or '/news/' in href:
                    # Make absolute URL
                    if href.startswith('/'):
                        href = self.news_sites['tribune']['base_url'] + href
                    elif not href.startswith('http'):
                        continue
                    
                    # Check if URL is relevant
                    if self._is_relevant_url(href):
                        article_links.append(href)
            
            # Remove duplicates and limit
            article_links = list(set(article_links))[:10]
            
            logging.info(f"[NewsEngine] Found {len(article_links)} Tribune article links")
            
            # Extract content from each article
            for article_url in article_links:
                article_data = self.extract_article(article_url, 'tribune')
                if article_data:
                    articles.append(article_data)
            
            self._update_stats('tribune_scraped')
            
        except Exception as e:
            logging.error(f"[NewsEngine] Error scraping Tribune: {e}")
            self._update_stats('errors')
        
        return articles
    
    def scrape_express(self, query: str = "Karachi mobile snatching") -> List[Dict]:
        """
        Scrape Express News for crime articles
        
        Args:
            query: Search query (default: "Karachi mobile snatching")
            
        Returns:
            List of article data dictionaries
        """
        logging.info(f"[NewsEngine] Scraping Express News for: {query}")
        
        articles = []
        
        try:
            # Build search URL
            search_url = self.news_sites['express']['search_url'].format(
                query=query.replace(' ', '+')
            )
            
            # Get search results page
            response = self.http_client.get(search_url, timeout=30)
            
            if not response or response.status_code != 200:
                logging.warning(f"[NewsEngine] Express search failed: status {response.status_code if response else 'None'}")
                self._update_stats('errors')
                return articles
            
            # Parse HTML to find article links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links
            article_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Express article URLs typically contain /story/ or /news/
                if '/story/' in href or '/news/' in href or '/epaper/' in href:
                    # Make absolute URL
                    if href.startswith('/'):
                        href = self.news_sites['express']['base_url'] + href
                    elif not href.startswith('http'):
                        continue
                    
                    # Check if URL is relevant
                    if self._is_relevant_url(href):
                        article_links.append(href)
            
            # Remove duplicates and limit
            article_links = list(set(article_links))[:10]
            
            logging.info(f"[NewsEngine] Found {len(article_links)} Express article links")
            
            # Extract content from each article
            for article_url in article_links:
                article_data = self.extract_article(article_url, 'express')
                if article_data:
                    articles.append(article_data)
            
            self._update_stats('express_scraped')
            
        except Exception as e:
            logging.error(f"[NewsEngine] Error scraping Express: {e}")
            self._update_stats('errors')
        
        return articles
    
    def extract_article(self, url: str, site_key: str) -> Optional[Dict]:
        """
        Extract full content from article URL
        
        Args:
            url: Article URL
            site_key: Site key ('dawn', 'geo', 'tribune', 'express')
            
        Returns:
            Dictionary with article data or None
        """
        # Check if already seen
        with self.seen_urls_lock:
            if url in self.seen_urls:
                logging.debug(f"[NewsEngine] Skipping duplicate URL: {url}")
                return None
            self.seen_urls.add(url)
        
        try:
            logging.debug(f"[NewsEngine] Extracting article: {url}")
            
            # Get article page
            response = self.http_client.get(url, timeout=30)
            
            if not response or response.status_code != 200:
                logging.warning(f"[NewsEngine] Failed to fetch article: {url}")
                return None
            
            # Parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = ''
            title_tags = soup.find_all(['h1', 'h2', 'title'])
            for tag in title_tags:
                if tag.get_text(strip=True):
                    title = tag.get_text(strip=True)
                    break
            
            # Extract article content using site-specific selectors
            content_parts = []
            selectors = self.news_sites[site_key]['article_selectors']
            
            for selector in selectors:
                # Try CSS selector
                if '.' in selector or '#' in selector:
                    elements = soup.select(selector)
                else:
                    # Try tag name
                    elements = soup.find_all(selector)
                
                for element in elements:
                    # Get all paragraph text
                    paragraphs = element.find_all('p')
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if len(text) > 20:  # Minimum length
                            content_parts.append(text)
                
                # If we found content, break
                if content_parts:
                    break
            
            # Combine content
            full_content = '\n\n'.join(content_parts)
            
            # Check if content is relevant
            if not self._is_relevant_content(title + ' ' + full_content):
                logging.debug(f"[NewsEngine] Article not relevant: {url}")
                return None
            
            # Build article data
            article_data = {
                'url': url,
                'source': self.news_sites[site_key]['name'],
                'site_key': site_key,
                'title': title,
                'content': full_content,
                'raw_text': f"{title}\n\n{full_content}",
                'timestamp': datetime.now().isoformat(),
                'credible': self.news_sites[site_key]['credible']
            }
            
            # Check for statistical data
            statistical_report = self.parse_statistics(full_content, url, site_key)
            if statistical_report:
                article_data['statistical_report'] = statistical_report
                self._update_stats('statistical_reports_found')
            
            self._update_stats('articles_extracted')
            logging.info(f"[NewsEngine] Extracted article from {self.news_sites[site_key]['name']}: {title[:50]}...")
            
            return article_data
            
        except Exception as e:
            logging.error(f"[NewsEngine] Error extracting article {url}: {e}")
            return None
    
    def parse_statistics(self, text: str, source_url: str, site_key: str) -> Optional[StatisticalReport]:
        """
        Detect and parse aggregate crime reports from article text
        
        Args:
            text: Article text to analyze
            source_url: URL of the article
            site_key: Site key for credibility verification
            
        Returns:
            StatisticalReport if statistics detected and source is credible, None otherwise
        """
        if not text or len(text) < 50:
            return None
        
        # Verify source credibility first (Requirement 20)
        if not self.verify_source(source_url):
            logging.debug(f"[NewsEngine] Source not credible for statistics: {source_url}")
            return None
        
        # Use statistical expander to detect statistics (Requirement 18)
        statistical_report = self.statistical_expander.detect_statistics(text)
        
        if statistical_report:
            logging.info(
                f"[NewsEngine] Found statistical report: {statistical_report.count} incidents "
                f"in {statistical_report.location or 'Karachi'}"
            )
            
            # Add source information
            statistical_report.source_url = source_url
            statistical_report.source_name = self.news_sites[site_key]['name']
        
        return statistical_report
    
    def verify_source(self, source_url: str) -> bool:
        """
        Verify source credibility for statistical data
        
        Args:
            source_url: URL of the source
            
        Returns:
            True if source is credible, False otherwise
        """
        # Use statistical expander's verify_source method (Requirement 20)
        return self.statistical_expander.verify_source(source_url)
    
    def scrape_all_sites(self, query: str = "Karachi mobile snatching") -> List[Dict]:
        """
        Scrape all news sites for the given query
        
        Args:
            query: Search query (default: "Karachi mobile snatching")
            
        Returns:
            Combined list of articles from all sites
        """
        logging.info(f"[NewsEngine] Scraping all sites for: {query}")
        
        all_articles = []
        
        # Scrape each site
        all_articles.extend(self.scrape_dawn(query))
        all_articles.extend(self.scrape_geo(query))
        all_articles.extend(self.scrape_tribune(query))
        all_articles.extend(self.scrape_express(query))
        
        logging.info(f"[NewsEngine] Total articles extracted: {len(all_articles)}")
        
        return all_articles
    
    def _is_relevant_url(self, url: str) -> bool:
        """
        Check if URL is relevant (contains Karachi and crime keywords)
        
        Args:
            url: URL to check
            
        Returns:
            True if URL appears relevant
        """
        url_lower = url.lower()
        
        # Check for Karachi mention
        has_karachi = any(keyword in url_lower for keyword in self.karachi_keywords)
        
        # Check for crime keywords
        has_crime = any(keyword in url_lower for keyword in self.crime_keywords)
        
        # URL is relevant if it mentions Karachi or crime
        return has_karachi or has_crime
    
    def _is_relevant_content(self, text: str) -> bool:
        """
        Check if content is relevant (Karachi crime-related)
        
        Args:
            text: Text to check
            
        Returns:
            True if content is relevant
        """
        if not text or len(text) < 50:
            return False
        
        text_lower = text.lower()
        
        # Must mention Karachi
        has_karachi = any(keyword in text_lower for keyword in self.karachi_keywords)
        
        if not has_karachi:
            return False
        
        # Must have crime keywords
        has_crime = any(keyword in text_lower for keyword in self.crime_keywords)
        
        return has_crime
    
    def _update_stats(self, key: str):
        """Update statistics"""
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + 1
    
    def get_stats(self) -> Dict:
        """
        Get scraping statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'dawn_scraped': 0,
                'geo_scraped': 0,
                'tribune_scraped': 0,
                'express_scraped': 0,
                'articles_extracted': 0,
                'statistical_reports_found': 0,
                'errors': 0
            }
            logging.info("[NewsEngine] Statistics reset")


# ============================================================================
# TASK 15: GOOGLE NEWS SCRAPING WITH DEEP CRAWLING
# ============================================================================

class GoogleNewsEngine:
    """
    Google News scraping engine with deep article crawling and anti-detection measures.
    
    Features:
    - Query-based search on Google News
    - Deep crawling to extract full article content from source websites
    - Result pagination handling
    - Anti-detection measures (user agent rotation, delays)
    - HTTP-based scraping with retry logic
    - Karachi crime-specific filtering
    
    Requirements addressed: 11, 27
    """
    
    def __init__(
        self,
        driver_manager: DriverPoolManager,
        http_client: HTTPClientPool,
        error_recovery: ErrorRecoverySystem
    ):
        """
        Initialize Google News engine
        
        Args:
            driver_manager: Driver pool manager for browser automation
            http_client: HTTP client pool for making requests
            error_recovery: Error recovery system for handling failures
        """
        self.driver_manager = driver_manager
        self.http_client = http_client
        self.error_recovery = error_recovery
        
        # Google News configuration
        self.base_url = "https://news.google.com"
        self.search_url = "https://news.google.com/search?q={query}&hl=en-PK&gl=PK&ceid=PK:en"
        
        # Crime-related keywords for filtering
        self.crime_keywords = [
            'snatch', 'theft', 'stolen', 'robbery', 'mugging', 'dacoity',
            'mobile', 'phone', 'crime', 'incident', 'loot', 'burglar',
            'street crime', 'armed robbery', 'motorcycle', 'valuables'
        ]
        
        # Karachi location keywords
        self.karachi_keywords = [
            'karachi', 'clifton', 'dha', 'gulshan', 'nazimabad', 'saddar',
            'malir', 'korangi', 'lyari', 'orangi', 'north nazimabad',
            'shah faisal', 'landhi', 'baldia', 'keamari', 'jamshed'
        ]
        
        # Statistics tracking
        self.stats = {
            'searches_performed': 0,
            'articles_found': 0,
            'articles_crawled': 0,
            'relevant_articles': 0,
            'errors': 0
        }
        self.stats_lock = threading.Lock()
        
        # Seen article URLs to avoid duplicates
        self.seen_urls = set()
        self.seen_urls_lock = threading.Lock()
        
        logging.info("[GoogleNewsEngine] Initialized with deep crawling support")
    
    def scrape_google_news(
        self,
        query: str = "Karachi mobile snatching",
        max_results: int = 20,
        deep_crawl: bool = True
    ) -> List[Dict]:
        """
        Scrape Google News for crime articles with query-based search
        
        Args:
            query: Search query (default: "Karachi mobile snatching")
            max_results: Maximum number of results to process (default: 20)
            deep_crawl: Whether to crawl full article content (default: True)
            
        Returns:
            List of article data dictionaries
        """
        logging.info(f"[GoogleNewsEngine] Scraping Google News for: {query} (max: {max_results})")
        
        articles = []
        driver = None
        
        try:
            # Get driver from pool
            driver = self.driver_manager.get_driver()
            
            # Build search URL
            search_url = self.search_url.format(query=query.replace(' ', '+'))
            
            # Navigate to search results
            logging.debug(f"[GoogleNewsEngine] Navigating to: {search_url}")
            driver.get(search_url)
            
            # Anti-detection: Random delay
            time.sleep(random.uniform(2, 4))
            
            # Scroll to load more results (pagination handling)
            self._scroll_to_load_more(driver)
            
            # Find article elements
            article_elements = driver.find_elements(By.TAG_NAME, 'article')
            
            logging.info(f"[GoogleNewsEngine] Found {len(article_elements)} article elements")
            
            # Process each article
            processed_count = 0
            for article_element in article_elements:
                if processed_count >= max_results:
                    break
                
                try:
                    # Extract article data from Google News card
                    article_data = self._extract_article_data(article_element, driver)
                    
                    if not article_data:
                        continue
                    
                    # Check if already seen
                    with self.seen_urls_lock:
                        if article_data['url'] in self.seen_urls:
                            logging.debug(f"[GoogleNewsEngine] Skipping duplicate: {article_data['url']}")
                            continue
                        self.seen_urls.add(article_data['url'])
                    
                    # Deep crawl to get full article content
                    if deep_crawl:
                        full_article = self.deep_crawl_article(article_data['url'])
                        if full_article:
                            # Merge deep crawl data with Google News data
                            article_data.update(full_article)
                            article_data['deep_crawled'] = True
                        else:
                            article_data['deep_crawled'] = False
                    else:
                        article_data['deep_crawled'] = False
                    
                    # Check relevance
                    if self._is_relevant_article(article_data):
                        articles.append(article_data)
                        self._update_stats('relevant_articles')
                        logging.info(
                            f"[GoogleNewsEngine] Extracted relevant article: "
                            f"{article_data.get('title', 'No title')[:50]}..."
                        )
                    
                    processed_count += 1
                    self._update_stats('articles_found')
                    
                    # Anti-detection: Small delay between articles
                    time.sleep(random.uniform(0.5, 1.5))
                    
                except Exception as e:
                    logging.warning(f"[GoogleNewsEngine] Error processing article element: {e}")
                    continue
            
            self._update_stats('searches_performed')
            logging.info(f"[GoogleNewsEngine] Extracted {len(articles)} relevant articles")
            
        except Exception as e:
            logging.error(f"[GoogleNewsEngine] Error scraping Google News: {e}")
            self._update_stats('errors')
            
        finally:
            # Return driver to pool
            if driver:
                self.driver_manager.release_driver(driver)
        
        return articles
    
    def _scroll_to_load_more(self, driver: webdriver.Chrome, max_scrolls: int = 3):
        """
        Scroll page to load more results (pagination handling)
        
        Args:
            driver: WebDriver instance
            max_scrolls: Maximum number of scroll attempts (default: 3)
        """
        logging.debug(f"[GoogleNewsEngine] Scrolling to load more results (max: {max_scrolls})")
        
        for i in range(max_scrolls):
            try:
                # Get current scroll height
                last_height = driver.execute_script("return document.body.scrollHeight")
                
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait for new content to load
                time.sleep(random.uniform(1.5, 2.5))
                
                # Calculate new scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                # Break if no new content loaded
                if new_height == last_height:
                    logging.debug(f"[GoogleNewsEngine] No more content to load after {i+1} scrolls")
                    break
                    
            except Exception as e:
                logging.warning(f"[GoogleNewsEngine] Error during scroll {i+1}: {e}")
                break
    
    def _extract_article_data(self, article_element, driver: webdriver.Chrome) -> Optional[Dict]:
        """
        Extract article data from Google News article element
        
        Args:
            article_element: Selenium WebElement for article
            driver: WebDriver instance
            
        Returns:
            Dictionary with article data or None
        """
        try:
            # Extract title
            title = ''
            try:
                title_element = article_element.find_element(By.TAG_NAME, 'h3')
                title = title_element.text.strip()
            except NoSuchElementException:
                try:
                    title_element = article_element.find_element(By.TAG_NAME, 'h4')
                    title = title_element.text.strip()
                except NoSuchElementException:
                    pass
            
            if not title:
                return None
            
            # Extract snippet/description
            snippet = ''
            try:
                # Google News uses various classes for snippets
                snippet_selectors = [
                    'div[class*="snippet"]',
                    'div[class*="description"]',
                    'span[class*="snippet"]'
                ]
                
                for selector in snippet_selectors:
                    try:
                        snippet_element = article_element.find_element(By.CSS_SELECTOR, selector)
                        snippet = snippet_element.text.strip()
                        if snippet:
                            break
                    except NoSuchElementException:
                        continue
                        
            except Exception:
                pass
            
            # Extract source
            source = ''
            try:
                source_element = article_element.find_element(By.CSS_SELECTOR, 'a[data-n-tid]')
                source = source_element.text.strip()
            except NoSuchElementException:
                source = 'Google News'
            
            # Extract article URL (the actual news site URL, not Google News URL)
            article_url = ''
            try:
                # Find the main link
                link_element = article_element.find_element(By.TAG_NAME, 'a')
                google_url = link_element.get_attribute('href')
                
                # Google News URLs are redirects, we need to extract the actual URL
                if google_url and 'articles/' in google_url:
                    # Try to get the actual URL by clicking and checking
                    # For now, we'll use the Google URL and resolve it later
                    article_url = google_url
                else:
                    article_url = google_url
                    
            except NoSuchElementException:
                return None
            
            if not article_url:
                return None
            
            # Extract timestamp
            timestamp = ''
            try:
                time_element = article_element.find_element(By.TAG_NAME, 'time')
                timestamp = time_element.get_attribute('datetime')
                if not timestamp:
                    timestamp = time_element.text.strip()
            except NoSuchElementException:
                timestamp = datetime.now().isoformat()
            
            # Build article data
            article_data = {
                'url': article_url,
                'title': title,
                'snippet': snippet,
                'source': source,
                'timestamp': timestamp,
                'raw_text': f"{title}\n\n{snippet}",
                'found_via': 'Google News'
            }
            
            return article_data
            
        except Exception as e:
            logging.debug(f"[GoogleNewsEngine] Error extracting article data: {e}")
            return None
    
    def deep_crawl_article(self, url: str) -> Optional[Dict]:
        """
        Deep crawl article to extract full content from source website
        
        Args:
            url: Article URL to crawl
            
        Returns:
            Dictionary with full article content or None
        """
        logging.debug(f"[GoogleNewsEngine] Deep crawling: {url}")
        
        try:
            # Resolve Google News redirect URL if needed
            if 'news.google.com' in url and '/articles/' in url:
                url = self._resolve_google_redirect(url)
                if not url:
                    logging.warning("[GoogleNewsEngine] Could not resolve Google News redirect")
                    return None
            
            # Get article page using HTTP client
            response = self.http_client.get(url, timeout=30)
            
            if not response or response.status_code != 200:
                logging.warning(f"[GoogleNewsEngine] Failed to fetch article: {url}")
                return None
            
            # Parse HTML
            if not HAS_BS4:
                logging.warning("[GoogleNewsEngine] BeautifulSoup not available, skipping deep crawl")
                return None
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract full content
            full_content = self._extract_full_content(soup)
            
            if not full_content or len(full_content) < 100:
                logging.debug(f"[GoogleNewsEngine] Insufficient content extracted from: {url}")
                return None
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            self._update_stats('articles_crawled')
            
            return {
                'full_content': full_content,
                'content_length': len(full_content),
                'metadata': metadata,
                'crawled_url': url
            }
            
        except Exception as e:
            logging.error(f"[GoogleNewsEngine] Error deep crawling {url}: {e}")
            return None
    
    def _resolve_google_redirect(self, google_url: str) -> Optional[str]:
        """
        Resolve Google News redirect URL to actual article URL
        
        Args:
            google_url: Google News URL
            
        Returns:
            Actual article URL or None
        """
        try:
            # Use HTTP client with allow_redirects to follow redirects
            response = self.http_client.get(google_url, timeout=10, allow_redirects=True)
            
            if response:
                # The final URL after redirects is the actual article URL
                actual_url = response.url
                logging.debug(f"[GoogleNewsEngine] Resolved redirect: {google_url} -> {actual_url}")
                return actual_url
            
        except Exception as e:
            logging.warning(f"[GoogleNewsEngine] Error resolving redirect: {e}")
        
        return None
    
    def _extract_full_content(self, soup: BeautifulSoup) -> str:
        """
        Extract full article content from parsed HTML
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            Full article content as string
        """
        content_parts = []
        
        # Common article content selectors (ordered by specificity)
        content_selectors = [
            'article',
            'div[class*="article-content"]',
            'div[class*="story-content"]',
            'div[class*="post-content"]',
            'div[class*="entry-content"]',
            'div[class*="content-body"]',
            'div[id*="article"]',
            'div[id*="content"]',
            'main',
            'div[role="main"]'
        ]
        
        # Try each selector
        for selector in content_selectors:
            try:
                elements = soup.select(selector)
                
                for element in elements:
                    # Extract all paragraphs
                    paragraphs = element.find_all('p')
                    
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        
                        # Filter out short paragraphs (likely navigation/ads)
                        if len(text) > 30:
                            content_parts.append(text)
                
                # If we found substantial content, break
                if len(content_parts) > 3:
                    break
                    
            except Exception as e:
                logging.debug(f"[GoogleNewsEngine] Error with selector {selector}: {e}")
                continue
        
        # If no content found with selectors, try all paragraphs
        if not content_parts:
            all_paragraphs = soup.find_all('p')
            for p in all_paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 30:
                    content_parts.append(text)
        
        # Combine content
        full_content = '\n\n'.join(content_parts)
        
        return full_content
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """
        Extract metadata from article HTML
        
        Args:
            soup: BeautifulSoup parsed HTML
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        try:
            # Extract author
            author_selectors = [
                'meta[name="author"]',
                'meta[property="article:author"]',
                'span[class*="author"]',
                'a[rel="author"]'
            ]
            
            for selector in author_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        if element.name == 'meta':
                            metadata['author'] = element.get('content', '')
                        else:
                            metadata['author'] = element.get_text(strip=True)
                        break
                except:
                    continue
            
            # Extract publish date
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publish-date"]',
                'time[datetime]',
                'span[class*="date"]'
            ]
            
            for selector in date_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        if element.name == 'meta':
                            metadata['published_date'] = element.get('content', '')
                        elif element.name == 'time':
                            metadata['published_date'] = element.get('datetime', element.get_text(strip=True))
                        else:
                            metadata['published_date'] = element.get_text(strip=True)
                        break
                except:
                    continue
            
            # Extract description
            desc_selectors = [
                'meta[name="description"]',
                'meta[property="og:description"]'
            ]
            
            for selector in desc_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        metadata['description'] = element.get('content', '')
                        break
                except:
                    continue
                    
        except Exception as e:
            logging.debug(f"[GoogleNewsEngine] Error extracting metadata: {e}")
        
        return metadata
    
    def _is_relevant_article(self, article_data: Dict) -> bool:
        """
        Check if article is relevant (Karachi crime-related)
        
        Args:
            article_data: Article data dictionary
            
        Returns:
            True if article is relevant
        """
        # Combine all text fields
        text_parts = [
            article_data.get('title', ''),
            article_data.get('snippet', ''),
            article_data.get('full_content', ''),
            article_data.get('raw_text', '')
        ]
        
        full_text = ' '.join(text_parts).lower()
        
        if len(full_text) < 50:
            return False
        
        # Must mention Karachi
        has_karachi = any(keyword in full_text for keyword in self.karachi_keywords)
        
        if not has_karachi:
            return False
        
        # Must have crime keywords
        has_crime = any(keyword in full_text for keyword in self.crime_keywords)
        
        return has_crime
    
    def _update_stats(self, key: str):
        """Update statistics"""
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + 1
    
    def get_stats(self) -> Dict:
        """
        Get scraping statistics
        
        Returns:
            Dictionary with statistics
        """
        with self.stats_lock:
            return dict(self.stats)
    
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'searches_performed': 0,
                'articles_found': 0,
                'articles_crawled': 0,
                'relevant_articles': 0,
                'errors': 0
            }
            logging.info("[GoogleNewsEngine] Statistics reset")


# ============================================================================
# FACEBOOK GRAPH API SCRAPER
# ============================================================================

class FacebookGraphAPIScraper:
    """
    Facebook Graph API scraper for public posts and pages
    Uses official Graph API to avoid rate limiting and account blocks
    """
    
    def __init__(self, config: Configuration, http_client: Optional[HTTPClientPool] = None):
        """
        Initialize Facebook Graph API scraper
        
        Args:
            config: Configuration with FB credentials
            http_client: Optional HTTP client pool
        """
        self.config = config
        self.access_token = config.FB_GRAPH_API_ACCESS_TOKEN
        self.api_url = config.FB_GRAPH_API_URL
        self.http_client = http_client or HTTPClientPool()
        self.enabled = config.has_facebook_graph_api()
        
        # Rate limiting
        self.requests_made = 0
        self.rate_limit_reached = False
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited': 0,
            'posts_found': 0,
            'errors': 0
        }
        
        if self.enabled:
            logging.info("[FacebookGraphAPI] Initialized with Graph API v18.0")
        else:
            logging.warning("[FacebookGraphAPI] Disabled - no access token configured")
    
    def search_public_posts(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Get posts from known public pages (Consumer apps can't use search endpoint)
        
        Args:
            query: Search query (used for filtering)
            limit: Maximum posts to retrieve
            
        Returns:
            List of post dictionaries
        """
        if not self.enabled or self.rate_limit_reached:
            return []
        
        posts = []
        
        # Consumer apps can only access specific pages/groups they have access to
        # These are public Karachi-related pages - you can add more page IDs here
        known_pages = [
            # Add real page IDs here - you can find them by:
            # 1. Go to facebook.com/[page-name]
            # 2. View page source and search for "page_id"
            # Or use: https://findmyfbid.com/
            
            # Example format:
            # {'id': 'PAGE_ID_HERE', 'name': 'Page Name'},
        ]
        
        # If no pages configured, log warning
        if not known_pages:
            logging.warning("[FacebookGraphAPI] No pages configured. Consumer apps need specific page IDs.")
            logging.info("[FacebookGraphAPI] To add pages:")
            logging.info("  1. Find page IDs using https://findmyfbid.com/")
            logging.info("  2. Add to known_pages list in FacebookGraphAPIScraper")
            logging.info("  3. Or use Selenium fallback scraper instead")
            return []
        
        logging.info(f"[FacebookGraphAPI] Fetching posts from {len(known_pages)} configured pages")
        
        for page_info in known_pages:
            if len(posts) >= limit:
                break
            
            page_posts = self._get_page_posts(
                page_info['id'], 
                page_info['name'], 
                limit=10
            )
            
            # Filter posts by query keywords
            query_keywords = query.lower().split()
            for post in page_posts:
                if any(keyword in post['text'].lower() for keyword in query_keywords):
                    posts.append(post)
        
        self.stats['posts_found'] += len(posts)
        
        if posts:
            logging.info(f"[FacebookGraphAPI] Total posts collected: {len(posts)}")
        else:
            logging.info("[FacebookGraphAPI] No posts found - consider using Selenium fallback")
        
        return posts[:limit]
    
    def _get_page_posts(self, page_id: str, page_name: str, limit: int = 10) -> List[Dict]:
        """
        Get posts from a specific page
        
        Args:
            page_id: Facebook page ID
            page_name: Page name for logging
            limit: Maximum posts to retrieve
            
        Returns:
            List of post dictionaries
        """
        posts = []
        
        try:
            self._apply_rate_limit()
            
            # Get page posts
            posts_url = f"{self.api_url}/{page_id}/posts"
            params = {
                'access_token': self.access_token,
                'fields': 'id,message,created_time,permalink_url',
                'limit': limit
            }
            
            self.stats['total_requests'] += 1
            response = self.http_client.get(
                url=posts_url,
                params=params,
                timeout=15
            )
            
            if response and response.status_code == 200:
                data = response.json()
                page_posts = data.get('data', [])
                
                logging.info(f"[FacebookGraphAPI] Found {len(page_posts)} posts from {page_name}")
                
                for post in page_posts:
                    message = post.get('message', '')
                    if message and len(message) > 20:
                        posts.append({
                            'text': message,
                            'url': post.get('permalink_url', ''),
                            'timestamp': post.get('created_time', ''),
                            'source': f"Facebook - {page_name}"
                        })
                
                self.stats['successful_requests'] += 1
            
            elif response and response.status_code == 429:
                logging.warning("[FacebookGraphAPI] Rate limit reached")
                self.rate_limit_reached = True
                self.stats['rate_limited'] += 1
        
        except Exception as e:
            logging.error(f"[FacebookGraphAPI] Error getting page posts: {e}")
            self.stats['errors'] += 1
        
        return posts
    
    def _apply_rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        return self.stats.copy()
    
    def reset_rate_limit(self):
        """Reset rate limit flag (call after waiting period)"""
        self.rate_limit_reached = False
        logging.info("[FacebookGraphAPI] Rate limit reset")


# ============================================================================
# TWITTER API SCRAPER
# ============================================================================

class TwitterAPIScraper:
    """
    Twitter API v2 scraper for crime-related tweets
    Uses official Twitter API with proper rate limiting
    """
    
    def __init__(self, config: Configuration, http_client: Optional[HTTPClientPool] = None):
        """
        Initialize Twitter API scraper
        
        Args:
            config: Configuration with Twitter credentials
            http_client: Optional HTTP client pool
        """
        self.config = config
        self.bearer_token = config.TWITTER_BEARER_TOKEN
        self.api_url = "https://api.twitter.com/2"
        self.http_client = http_client or HTTPClientPool()
        self.enabled = config.has_twitter_api()
        
        # Rate limiting (Twitter: 450 requests per 15 min = 30 per minute)
        self.requests_made = 0
        self.rate_limit_reached = False
        self.rate_limit_reset_time = 0
        self.requests_per_window = 25  # Conservative limit
        self.window_start_time = time.time()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited': 0,
            'tweets_found': 0,
            'errors': 0
        }
        
        if self.enabled:
            logging.info("[TwitterAPI] Initialized with Twitter API v2")
        else:
            logging.warning("[TwitterAPI] Disabled - no bearer token configured")
    
    def search_tweets(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search recent tweets using Twitter API v2 with improved rate limiting
        
        Args:
            query: Search query
            max_results: Maximum tweets to retrieve (10-100)
            
        Returns:
            List of tweet dictionaries
        """
        if not self.enabled or self.rate_limit_reached:
            if self.rate_limit_reached:
                logging.warning("[TwitterAPI] Rate limit active, skipping request")
            return []
        
        tweets = []
        
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                return []
            
            # Build query with Karachi context - try multiple variations
            query_variations = [
                f"{query} Karachi -is:retweet lang:en",
                f"{query} Karachi -is:retweet",
                f"Karachi {query} -is:retweet"
            ]
            
            for query_variant in query_variations:
                if len(tweets) >= max_results:
                    break
                
                # Check rate limit before each request
                if not self._check_rate_limit():
                    break
                
                # Search tweets
                search_url = f"{self.api_url}/tweets/search/recent"
                params = {
                    'query': query_variant,
                    'max_results': min(max_results, 100),  # API limit is 100
                    'tweet.fields': 'created_at,text,author_id',
                    'expansions': 'author_id',
                    'user.fields': 'username'
                }
                
                headers = {
                    'Authorization': f'Bearer {self.bearer_token}'
                }
                
                self.stats['total_requests'] += 1
                self.requests_made += 1
                
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
                
                response = self.http_client.get(
                    url=search_url,
                    params=params,
                    headers=headers,
                    timeout=15
                )
                
                if response and response.status_code == 200:
                    data = response.json()
                    tweet_data = data.get('data', [])
                    
                    if not tweet_data:
                        logging.debug(f"[TwitterAPI] No tweets for variant: {query_variant}")
                        continue
                    
                    users = {u['id']: u for u in data.get('includes', {}).get('users', [])}
                    
                    logging.info(f"[TwitterAPI] Found {len(tweet_data)} tweets for query: {query}")
                    
                    for tweet in tweet_data:
                        text = tweet.get('text', '')
                        author_id = tweet.get('author_id', '')
                        username = users.get(author_id, {}).get('username', 'unknown')
                        
                        if text and len(text) > 20:
                            tweets.append({
                                'text': text,
                                'url': f"https://twitter.com/{username}/status/{tweet['id']}",
                                'timestamp': tweet.get('created_at', ''),
                                'source': f"Twitter - @{username}"
                            })
                    
                    self.stats['successful_requests'] += 1
                    self.stats['tweets_found'] += len(tweet_data)
                    
                    # If we got results, no need to try other variants
                    if tweet_data:
                        break
                
                elif response and response.status_code == 429:
                    logging.warning("[TwitterAPI] Rate limit reached - will skip Twitter for remaining queries")
                    self._handle_rate_limit(response)
                    self.rate_limit_reached = True  # Mark as rate-limited to skip future requests
                    break  # Stop trying variants
                
                else:
                    error_msg = "No response"
                    if response:
                        error_msg = f"{response.status_code}"
                        try:
                            error_data = response.json()
                            if 'errors' in error_data:
                                error_msg += f" - {error_data['errors'][0].get('message', '')}"
                        except:
                            pass
                    
                    logging.warning(f"[TwitterAPI] Search failed: {error_msg}")
                    self.stats['errors'] += 1
        
        except Exception as e:
            logging.error(f"[TwitterAPI] Error searching tweets: {e}")
            self.stats['errors'] += 1
        
        return tweets
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits
        
        Returns:
            True if we can make request, False if rate limited
        """
        current_time = time.time()
        
        # Reset window if 15 minutes passed
        if current_time - self.window_start_time > 900:  # 15 minutes
            self.requests_made = 0
            self.window_start_time = current_time
            self.rate_limit_reached = False
        
        # Check if we've hit our limit
        if self.requests_made >= self.requests_per_window:
            self.rate_limit_reached = True
            wait_time = 900 - (current_time - self.window_start_time)
            logging.warning(f"[TwitterAPI] Rate limit reached. Wait {wait_time:.0f}s before next request")
            return False
        
        return True
    
    def _handle_rate_limit(self, response):
        """Handle rate limit response from Twitter"""
        self.rate_limit_reached = True
        self.stats['rate_limited'] += 1
        
        # Try to get reset time from headers
        if response and hasattr(response, 'headers'):
            reset_time = response.headers.get('x-rate-limit-reset')
            if reset_time:
                self.rate_limit_reset_time = int(reset_time)
                wait_time = self.rate_limit_reset_time - time.time()
                logging.warning(f"[TwitterAPI] Rate limit will reset in {wait_time:.0f}s")
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        return self.stats.copy()
    
    def reset_rate_limit(self):
        """Reset rate limit flag (call after waiting period)"""
        current_time = time.time()
        if current_time >= self.rate_limit_reset_time:
            self.rate_limit_reached = False
            self.requests_made = 0
            self.window_start_time = current_time
            logging.info("[TwitterAPI] Rate limit reset")


# ============================================================================
# SELENIUM FALLBACK SCRAPERS
# ============================================================================

class FacebookSeleniumScraper:
    """Facebook scraper using Selenium with optional authentication"""
    
    def __init__(self, email: str = None, password: str = None):
        self.driver = None
        self.email = email
        self.password = password
        self.is_logged_in = False
        self.scraped_count = 0
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'posts_found': 0,
            'errors': 0,
            'login_attempts': 0,
            'login_successful': False
        }
        
    def init_driver(self):
        """Initialize Selenium driver"""
        if self.driver:
            return
        
        options = Options()
        # Don't use headless if we need to login (helps with detection)
        if not self.email:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Add preferences to save login session
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.notifications": 2
        })
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logging.info("[FacebookSelenium] Driver initialized")
            
            # Login if credentials provided
            if self.email and self.password and not self.is_logged_in:
                self.login()
        except Exception as e:
            logging.error(f"[FacebookSelenium] Failed to initialize driver: {e}")
            raise
    
    def login(self):
        """Login to Facebook"""
        if self.is_logged_in:
            return True
        
        if not self.email or not self.password:
            logging.warning("[FacebookSelenium] No credentials provided, skipping login")
            return False
        
        try:
            self.stats['login_attempts'] += 1
            logging.info("[FacebookSelenium] Attempting to login...")
            
            # Go to Facebook
            self.driver.get("https://www.facebook.com/")
            time.sleep(3)
            
            # Find and fill email
            try:
                email_field = self.driver.find_element(By.ID, "email")
                email_field.clear()
                email_field.send_keys(self.email)
                logging.debug("[FacebookSelenium] Email entered")
            except:
                # Try alternative selector
                email_field = self.driver.find_element(By.NAME, "email")
                email_field.clear()
                email_field.send_keys(self.email)
            
            time.sleep(1)
            
            # Find and fill password
            try:
                password_field = self.driver.find_element(By.ID, "pass")
                password_field.clear()
                password_field.send_keys(self.password)
                logging.debug("[FacebookSelenium] Password entered")
            except:
                # Try alternative selector
                password_field = self.driver.find_element(By.NAME, "pass")
                password_field.clear()
                password_field.send_keys(self.password)
            
            time.sleep(1)
            
            # Click login button
            try:
                login_button = self.driver.find_element(By.NAME, "login")
                login_button.click()
            except:
                # Try alternative selector
                login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                login_button.click()
            
            logging.info("[FacebookSelenium] Login button clicked, waiting for response...")
            time.sleep(5)
            
            # Check if login successful
            current_url = self.driver.current_url
            if "login" not in current_url.lower() and "facebook.com" in current_url:
                self.is_logged_in = True
                self.stats['login_successful'] = True
                logging.info("[FacebookSelenium] ✓ Login successful!")
                
                # Handle any post-login popups
                try:
                    # Close "Save Login Info" popup if appears
                    not_now_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Not Now')]")
                    if not_now_buttons:
                        not_now_buttons[0].click()
                        time.sleep(1)
                except:
                    pass
                
                return True
            else:
                logging.error("[FacebookSelenium] Login failed - still on login page")
                return False
        
        except Exception as e:
            logging.error(f"[FacebookSelenium] Login error: {e}")
            return False
    
    def scrape_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Scrape Facebook search results (requires login)
        """
        posts = []
        self.stats['total_requests'] += 1
        
        if not self.is_logged_in:
            logging.warning("[FacebookSelenium] Not logged in, cannot search")
            return []
        
        try:
            # Check if driver is still alive
            try:
                _ = self.driver.current_url
            except:
                logging.warning("[FacebookSelenium] Driver crashed, reinitializing...")
                self.driver = None
                self.is_logged_in = False
                self.init_driver()
                if not self.is_logged_in:
                    return []
            
            # Use Facebook search - try both old and new URL formats
            search_query = f"{query} Karachi"
            # New Facebook search URL format
            search_url = f"https://www.facebook.com/search/posts/?q={search_query.replace(' ', '%20')}"
            
            logging.info(f"[FacebookSelenium] Searching: {search_query}")
            self.driver.get(search_url)
            time.sleep(6)  # Give more time for Facebook to load
            
            # Check if we're still logged in
            current_url = self.driver.current_url
            if 'login' in current_url.lower():
                logging.error("[FacebookSelenium] Redirected to login page - session expired")
                self.is_logged_in = False
                return []
            
            # Scroll to load more results
            for scroll in range(5):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                logging.debug(f"[FacebookSelenium] Scroll {scroll + 1}/5")
            
            # Modern Facebook uses different selectors - try multiple approaches
            seen_texts = set()
            
            # Strategy 1: Find article containers (most reliable for posts)
            try:
                articles = self.driver.find_elements(By.CSS_SELECTOR, 'div[role="article"]')
                logging.info(f"[FacebookSelenium] Found {len(articles)} article containers")
                
                for article in articles[:limit * 2]:
                    try:
                        # Get all text from the article
                        text = article.text.strip()
                        
                        # Skip if too short or already seen
                        if len(text) < 50 or text in seen_texts:
                            continue
                        
                        # Check for crime keywords
                        crime_keywords = ['snatch', 'theft', 'robbery', 'stolen', 'mobile', 'phone', 'crime', 'loot', 'چوری', 'لوٹ', 'چھین']
                        karachi_keywords = ['karachi', 'کراچی']
                        
                        text_lower = text.lower()
                        has_crime = any(keyword in text_lower for keyword in crime_keywords)
                        has_karachi = any(keyword in text_lower for keyword in karachi_keywords)
                        
                        if has_crime and has_karachi:
                            posts.append({
                                'text': text,
                                'url': search_url,
                                'timestamp': datetime.now().isoformat(),
                                'source': 'Facebook Search (Selenium)'
                            })
                            seen_texts.add(text)
                            self.scraped_count += 1
                            logging.info(f"[FacebookSelenium] Found: {text[:80]}...")
                            
                            if len(posts) >= limit:
                                break
                    
                    except Exception as e:
                        logging.debug(f"[FacebookSelenium] Error extracting from article: {e}")
                        continue
            
            except Exception as e:
                logging.warning(f"[FacebookSelenium] Error finding articles: {e}")
            
            # Strategy 2: If no articles found, try finding text elements directly
            if not posts:
                logging.info("[FacebookSelenium] No articles found, trying direct text elements")
                
                text_selectors = [
                    'div[data-ad-comet-preview="message"]',
                    'div[data-ad-preview="message"]',
                    'div[dir="auto"][style*="text-align"]',
                    'span[dir="auto"]'
                ]
                
                for selector in text_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            logging.info(f"[FacebookSelenium] Found {len(elements)} elements with {selector}")
                            
                            for element in elements[:limit * 3]:
                                try:
                                    text = element.text.strip()
                                    
                                    if len(text) > 50 and text not in seen_texts:
                                        crime_keywords = ['snatch', 'theft', 'robbery', 'stolen', 'mobile', 'phone', 'crime', 'loot', 'چوری', 'لوٹ']
                                        karachi_keywords = ['karachi', 'کراچی']
                                        
                                        text_lower = text.lower()
                                        has_crime = any(keyword in text_lower for keyword in crime_keywords)
                                        has_karachi = any(keyword in text_lower for keyword in karachi_keywords)
                                        
                                        if has_crime and has_karachi:
                                            posts.append({
                                                'text': text,
                                                'url': search_url,
                                                'timestamp': datetime.now().isoformat(),
                                                'source': 'Facebook Search (Selenium)'
                                            })
                                            seen_texts.add(text)
                                            self.scraped_count += 1
                                            logging.debug(f"[FacebookSelenium] Found: {text[:80]}...")
                                            
                                            if len(posts) >= limit:
                                                break
                                
                                except Exception as e:
                                    logging.debug(f"[FacebookSelenium] Error extracting text: {e}")
                                    continue
                            
                            if posts:
                                break  # Found posts with this selector, no need to try others
                    
                    except Exception as e:
                        logging.debug(f"[FacebookSelenium] Error with selector {selector}: {e}")
                        continue
            
            if posts:
                self.stats['successful_requests'] += 1
                self.stats['posts_found'] += len(posts)
                logging.info(f"[FacebookSelenium] Found {len(posts)} posts from search")
            else:
                logging.warning("[FacebookSelenium] No posts found in search")
                self.stats['errors'] += 1
        
        except Exception as e:
            logging.error(f"[FacebookSelenium] Search error: {e}")
            self.stats['errors'] += 1
        
        return posts
    
    def scrape_public_pages(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Scrape public Facebook pages for Karachi crime content
        Works with or without login
        """
        posts = []
        self.stats['total_requests'] += 1
        
        try:
            self.init_driver()
            
            # Strategy: Use public Karachi news/crime pages
            public_pages = [
                "https://www.facebook.com/KarachiUpdatesOfficial",
                "https://www.facebook.com/KarachiNewsOfficial",
                "https://www.facebook.com/DawnDotCom",
                "https://www.facebook.com/GeoUrdu",
            ]
            
            for page_url in public_pages:
                if len(posts) >= limit:
                    break
                
                try:
                    logging.info(f"[FacebookSelenium] Trying page: {page_url}")
                    self.driver.get(page_url)
                    time.sleep(4)
                    
                    # Scroll to load posts
                    for _ in range(3):
                        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)
                    
                    # Get page source and extract text
                    page_source = self.driver.page_source
                    
                    # Find post content - try multiple selectors
                    post_selectors = [
                        'div[data-ad-preview="message"]',
                        'div[data-ad-comet-preview="message"]',
                        'div[dir="auto"]',
                        'span[dir="auto"]'
                    ]
                    
                    post_elements = []
                    for selector in post_selectors:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            post_elements = elements
                            logging.info(f"[FacebookSelenium] Found {len(elements)} elements with selector: {selector}")
                            break
                    
                    seen_texts = set()
                    for element in post_elements[:limit * 2]:  # Get more to filter
                        try:
                            text = element.text.strip()
                            
                            # Filter for relevant content
                            if len(text) > 50 and text not in seen_texts:
                                # Check if it's Karachi crime-related
                                karachi_keywords = ['karachi', 'کراچی']
                                crime_keywords = ['snatch', 'theft', 'robbery', 'stolen', 'mobile', 'phone', 'crime', 'police', 'چوری', 'لوٹ']
                                
                                has_karachi = any(keyword in text.lower() for keyword in karachi_keywords)
                                has_crime = any(keyword in text.lower() for keyword in crime_keywords)
                                
                                if has_karachi and has_crime:
                                    posts.append({
                                        'text': text,
                                        'url': page_url,
                                        'timestamp': datetime.now().isoformat(),
                                        'source': 'Facebook Page (Selenium)'
                                    })
                                    seen_texts.add(text)
                                    self.scraped_count += 1
                                    logging.debug(f"[FacebookSelenium] Found relevant post: {text[:100]}...")
                                    
                                    if len(posts) >= limit:
                                        break
                        
                        except Exception as e:
                            logging.debug(f"[FacebookSelenium] Error extracting post: {e}")
                            continue
                
                except Exception as e:
                    logging.warning(f"[FacebookSelenium] Error accessing page {page_url}: {e}")
                    continue
            
            if posts:
                self.stats['successful_requests'] += 1
                self.stats['posts_found'] += len(posts)
                logging.info(f"[FacebookSelenium] Successfully extracted {len(posts)} posts")
            else:
                logging.warning("[FacebookSelenium] No relevant posts found on public pages")
                self.stats['errors'] += 1
        
        except Exception as e:
            logging.error(f"[FacebookSelenium] Scraping error: {e}")
            self.stats['errors'] += 1
        
        return posts
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        return self.stats.copy()
    
    def close(self):
        """Close driver"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            logging.info("[FacebookSelenium] Driver closed")


class TwitterSeleniumScraper:
    """Twitter scraper using Selenium with optional authentication"""
    
    def __init__(self, username: str = None, password: str = None):
        self.driver = None
        self.username = username
        self.password = password
        self.is_logged_in = False
        self.scraped_count = 0
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'tweets_found': 0,
            'errors': 0,
            'login_attempts': 0,
            'login_successful': False
        }
        
    def init_driver(self):
        """Initialize Selenium driver"""
        if self.driver:
            return
        
        options = Options()
        # Don't use headless if we need to login
        if not self.username:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-notifications')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logging.info("[TwitterSelenium] Driver initialized")
            
            # Login if credentials provided
            if self.username and self.password and not self.is_logged_in:
                self.login()
        except Exception as e:
            logging.error(f"[TwitterSelenium] Failed to initialize driver: {e}")
            raise
    
    def login(self):
        """Login to Twitter"""
        if self.is_logged_in:
            return True
        
        if not self.username or not self.password:
            logging.warning("[TwitterSelenium] No credentials provided, skipping login")
            return False
        
        try:
            self.stats['login_attempts'] += 1
            logging.info("[TwitterSelenium] Attempting to login...")
            
            # Go to Twitter login
            self.driver.get("https://twitter.com/i/flow/login")
            time.sleep(4)
            
            # Enter username
            try:
                username_field = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[autocomplete='username']"))
                )
                username_field.clear()
                username_field.send_keys(self.username)
                logging.debug("[TwitterSelenium] Username entered")
                time.sleep(1)
                
                # Click Next
                next_button = self.driver.find_element(By.XPATH, "//span[text()='Next']")
                next_button.click()
                time.sleep(3)
            except Exception as e:
                logging.error(f"[TwitterSelenium] Error entering username: {e}")
                return False
            
            # Enter password
            try:
                password_field = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='password']"))
                )
                password_field.clear()
                password_field.send_keys(self.password)
                logging.debug("[TwitterSelenium] Password entered")
                time.sleep(1)
                
                # Click Login
                login_button = self.driver.find_element(By.XPATH, "//span[text()='Log in']")
                login_button.click()
                logging.info("[TwitterSelenium] Login button clicked, waiting for response...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"[TwitterSelenium] Error entering password: {e}")
                return False
            
            # Check if login successful
            current_url = self.driver.current_url
            if "home" in current_url.lower() or "twitter.com" in current_url and "login" not in current_url:
                self.is_logged_in = True
                self.stats['login_successful'] = True
                logging.info("[TwitterSelenium] ✓ Login successful!")
                return True
            else:
                logging.error("[TwitterSelenium] Login failed - still on login page")
                return False
        
        except Exception as e:
            logging.error(f"[TwitterSelenium] Login error: {e}")
            return False
    
    def scrape_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Scrape Twitter search results (public, no login required)
        Uses multiple strategies to extract tweet content
        """
        tweets = []
        self.stats['total_requests'] += 1
        
        try:
            self.init_driver()
            
            # Build search query
            search_query = f"{query} Karachi -filter:retweets"
            search_url = f"https://twitter.com/search?q={search_query.replace(' ', '%20')}&src=typed_query&f=live"
            
            logging.info(f"[TwitterSelenium] Scraping: {search_url}")
            self.driver.get(search_url)
            time.sleep(5)  # Wait for initial load
            
            # Scroll to load more tweets
            for scroll in range(4):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                logging.debug(f"[TwitterSelenium] Scroll {scroll + 1}/4")
            
            # Strategy 1: Try to find tweet articles first
            tweet_articles = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            
            if tweet_articles:
                logging.info(f"[TwitterSelenium] Found {len(tweet_articles)} tweet articles")
                
                seen_texts = set()
                for article in tweet_articles[:limit * 2]:
                    try:
                        # Try to find tweet text within article
                        try:
                            text_element = article.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
                            text = text_element.text.strip()
                        except:
                            # Fallback: get all text from article
                            text = article.text.strip()
                            # Remove common Twitter UI text
                            for ui_text in ['Show more', 'Show less', 'Translate post', 'Replying to']:
                                text = text.replace(ui_text, '')
                        
                        # Filter and validate
                        if len(text) > 30 and text not in seen_texts:
                            # Check for crime-related keywords
                            crime_keywords = ['snatch', 'theft', 'robbery', 'stolen', 'mobile', 'phone', 'crime', 'loot']
                            if any(keyword in text.lower() for keyword in crime_keywords):
                                tweets.append({
                                    'text': text,
                                    'url': search_url,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'Twitter (Selenium)'
                                })
                                seen_texts.add(text)
                                self.scraped_count += 1
                                logging.debug(f"[TwitterSelenium] Found tweet: {text[:80]}...")
                                
                                if len(tweets) >= limit:
                                    break
                    
                    except Exception as e:
                        logging.debug(f"[TwitterSelenium] Error extracting from article: {e}")
                        continue
            
            # Strategy 2: If no articles found, try direct text elements
            if not tweets:
                logging.info("[TwitterSelenium] No articles found, trying direct text elements")
                
                text_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[lang]')
                logging.info(f"[TwitterSelenium] Found {len(text_elements)} text elements")
                
                seen_texts = set()
                for element in text_elements[:limit * 3]:
                    try:
                        text = element.text.strip()
                        
                        if len(text) > 30 and text not in seen_texts:
                            crime_keywords = ['snatch', 'theft', 'robbery', 'stolen', 'mobile', 'phone', 'crime', 'loot']
                            karachi_keywords = ['karachi', 'کراچی']
                            
                            has_crime = any(keyword in text.lower() for keyword in crime_keywords)
                            has_karachi = any(keyword in text.lower() for keyword in karachi_keywords)
                            
                            if has_crime and has_karachi:
                                tweets.append({
                                    'text': text,
                                    'url': search_url,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'Twitter (Selenium)'
                                })
                                seen_texts.add(text)
                                self.scraped_count += 1
                                logging.debug(f"[TwitterSelenium] Found tweet: {text[:80]}...")
                                
                                if len(tweets) >= limit:
                                    break
                    
                    except Exception as e:
                        logging.debug(f"[TwitterSelenium] Error extracting text: {e}")
                        continue
            
            if not tweets:
                logging.warning("[TwitterSelenium] No tweet elements found - Twitter may require login or changed structure")
                self.stats['errors'] += 1
            
            if tweets:
                self.stats['successful_requests'] += 1
                self.stats['tweets_found'] += len(tweets)
                logging.info(f"[TwitterSelenium] Successfully extracted {len(tweets)} tweets")
            else:
                logging.warning("[TwitterSelenium] No relevant tweets found")
                self.stats['errors'] += 1
        
        except Exception as e:
            logging.error(f"[TwitterSelenium] Scraping error: {e}")
            self.stats['errors'] += 1
        
        return tweets
    
    def get_statistics(self) -> Dict:
        """Get scraping statistics"""
        return self.stats.copy()
    
    def close(self):
        """Close driver"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            logging.info("[TwitterSelenium] Driver closed")


class UltimateScraper:
    """Ultimate scraper with deep crawling"""
    
    def __init__(self, config: Configuration, knowledge_base: Optional['KnowledgeBase'] = None):
        self.config = config
        self.driver = None
        self.knowledge_base = knowledge_base  # For Facebook auto-discovery
        
        # Create HTTP client pool
        self.http_client = HTTPClientPool()
        
        # Create LLMService if any API key is configured
        llm_service = None
        if config.has_any_llm_api():
            llm_service = LLMService(
                api_key=config.OPENROUTER_API_KEY,
                groq_api_key=config.GROQ_API_KEY,
                gemini_api_key=config.GEMINI_API_KEY,
                cerebras_api_key=config.CEREBRAS_API_KEY,
                cerebras_api_key2=config.CEREBRAS_API_KEY2,
                huggingface_api_key=config.HUGGING_FACE_API_KEY,
                huggingface_model_id=config.HUGGING_FACE_MODEL_ID,
                chatgpt_api_key=config.CHAT_GPT_API_KEY,
                deepseek_api_key=config.DEEPSEEK_API_KEY,
                http_client=self.http_client,
                test_mode=config.LLM_TEST_MODE
            )
        self.nlp = AdvancedNLP(llm_service)
        self.scraped_ids = set()
        
        # Initialize API scrapers
        self.fb_graph_scraper = FacebookGraphAPIScraper(config, self.http_client)
        self.twitter_scraper = TwitterAPIScraper(config, self.http_client)
        
        # Initialize Selenium fallback scrapers
        self.fb_selenium_scraper = FacebookSeleniumScraper()
        self.twitter_selenium_scraper = TwitterSeleniumScraper()
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    def init_driver(self):
        if self.driver:
            return
        
        options = Options()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(f'user-agent={random.choice(self.user_agents)}')
        options.add_argument('--disable-cache')
        options.add_argument('--disk-cache-size=1')
        
        # Memory and stability improvements to prevent tab crashes
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-background-networking')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-features=TranslateUI')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--single-process')
        options.add_argument('--aggressive-cache-discard')
        options.add_argument('--disable-application-cache')
        
        options.page_load_strategy = 'normal'
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def close_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
    
    def restart_driver(self):
        """Restart driver when it crashes"""
        logger.info("  [DRIVER] Restarting driver...")
        self.close_driver()
        time.sleep(2)
        self.init_driver()
        logger.info("  [DRIVER] Driver restarted successfully")
    
    def scrape_google_news(self, query: str) -> List[CrimeIncident]:
        """Scrape Google News with deep article crawling"""
        incidents = []
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.init_driver()
                url = f"https://news.google.com/search?q={query.replace(' ', '+')}&hl=en-PK&gl=PK&ceid=PK:en"
                
                self.driver.get(url)
                time.sleep(2)
                
                # Scroll to load more
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                articles = self.driver.find_elements(By.TAG_NAME, 'article')
                
                for article in articles[:15]:
                    try:
                        text = article.text
                        
                        if len(text) < 25:
                            continue
                        
                        text_id = hashlib.md5(text.encode()).hexdigest()
                        if text_id in self.scraped_ids:
                            continue
                        
                        incident = self.nlp.process(text, 'Google News', url)
                        
                        if incident:
                            incidents.append(incident)
                            self.scraped_ids.add(text_id)
                    
                    except:
                        continue
                
                break  # Success, exit retry loop
            
            except WebDriverException as e:
                logger.error(f"Google News driver error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.restart_driver()
                else:
                    logger.error("Google News: Max retries reached")
            
            except Exception as e:
                logger.error(f"Google News error: {e}")
                break
        
        return incidents
    
    def scrape_facebook_groups(self, query: str) -> List[CrimeIncident]:
        """Scrape specific Facebook groups for Karachi crime"""
        incidents = []
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                self.init_driver()
                
                for group_url in self.config.FACEBOOK_GROUPS:
                    try:
                        logger.info(f"  [FB Group] Scraping {group_url}")
                        
                        self.driver.get(group_url)
                        time.sleep(5)
                        
                        # Scroll to load more posts
                        for _ in range(5):
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(2)
                        
                        # Find all post containers with multiple selectors
                        post_selectors = [
                            'div[role="article"]',
                            'div[data-ad-preview="message"]',
                            'div.userContent',
                            'div._5pbx',
                            'span[dir="auto"]',
                            'div[data-testid="post_message"]',
                            'div.kvgmc6g5.cxmmr5t8.oygrvhab.hcukyx3x.c1et5uql'
                        ]
                        
                        all_posts = []
                        for selector in post_selectors:
                            try:
                                posts = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                all_posts.extend(posts)
                            except:
                                continue
                        
                        logger.info(f"  [FB Group] Found {len(all_posts)} potential posts")
                        
                        for post in all_posts[:30]:
                            try:
                                text = post.text
                                
                                if len(text) < 20:
                                    continue
                                
                                text_id = hashlib.md5(text.encode()).hexdigest()
                                if text_id in self.scraped_ids:
                                    continue
                                
                                incident = self.nlp.process(text, f'Facebook Group', group_url)
                                
                                if incident:
                                    incidents.append(incident)
                                    self.scraped_ids.add(text_id)
                                    logger.info(f"  [FB Group] Found incident in {incident.area}")
                            
                            except:
                                continue
                    
                    except Exception as e:
                        logger.error(f"  [FB Group] Error: {e}")
                        continue
                
                break  # Success, exit retry loop
            
            except WebDriverException as e:
                logger.error(f"Facebook groups driver error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.restart_driver()
                else:
                    logger.error("Facebook groups: Max retries reached")
            
            except Exception as e:
                logger.error(f"Facebook groups error: {e}")
                break
        
        return incidents
    
    def scrape_facebook_public(self, query: str) -> List[CrimeIncident]:
        """
        Scrape Facebook public posts with AUTO-DISCOVERY
        Uses Graph API first, falls back to Selenium if needed
        
        Features:
        - Searches Facebook for crime-related posts
        - Discovers new pages/groups posting crime reports
        - Saves discovered sources to knowledge base
        - Uses discovered sources in future runs
        """
        incidents = []
        discovered_sources = {}  # Track sources found in this run
        
        # Try Graph API first (faster, no rate limit issues)
        if self.fb_graph_scraper.enabled and not self.fb_graph_scraper.rate_limit_reached:
            logger.info(f"  [FB] Using Graph API for query: {query}")
            posts = self.fb_graph_scraper.search_public_posts(query, limit=20)
            
            for post_data in posts:
                try:
                    text = post_data.get('text', '')
                    url = post_data.get('url', '')
                    timestamp = post_data.get('timestamp', '')
                    source = post_data.get('source', 'Facebook')
                    
                    if len(text) < 25:
                        continue
                    
                    text_id = hashlib.md5(text.encode()).hexdigest()
                    if text_id in self.scraped_ids:
                        continue
                    
                    incident = self.nlp.process_text(text, source, url, timestamp)
                    
                    if incident:
                        incidents.append(incident)
                        self.scraped_ids.add(text_id)
                        logger.info(f"  [FB-GraphAPI] Found incident: {incident.area} - {incident.incident_type}")
                
                except Exception as e:
                    logger.debug(f"  [FB-GraphAPI] Error processing post: {e}")
                    continue
            
            if incidents:
                logger.info(f"  [FB-GraphAPI] Found {len(incidents)} incidents via Graph API")
                return incidents
            else:
                logger.info(f"  [FB-GraphAPI] No incidents found, falling back to Selenium")
        
        # Fallback to Selenium scraping if Graph API didn't work or is rate limited
        try:
            self.init_driver()
            
            # Step 1: Search Facebook for the query
            search_url = f"https://www.facebook.com/search/posts/?q={query.replace(' ', '%20')}"
            logger.info(f"  [FB] Searching: {search_url}")
            
            self.driver.get(search_url)
            time.sleep(5)
            
            # Scroll to load more posts
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Find all post containers
            post_selectors = [
                'div[role="article"]',
                'div[data-ad-preview="message"]',
                'div[data-testid="post_message"]',
                'div.userContent',
                'span[dir="auto"]'
            ]
            
            for selector in post_selectors:
                try:
                    posts = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    logger.info(f"  [FB] Found {len(posts)} posts with selector: {selector}")
                    
                    for post in posts[:20]:
                        try:
                            text = post.text
                            
                            if len(text) < 25:
                                continue
                            
                            text_id = hashlib.md5(text.encode()).hexdigest()
                            if text_id in self.scraped_ids:
                                continue
                            
                            # Try to extract source page/group URL from post
                            source_url = None
                            try:
                                # Look for profile/page links in the post
                                links = post.find_elements(By.TAG_NAME, 'a')
                                for link in links:
                                    href = link.get_attribute('href')
                                    if href and ('facebook.com/' in href):
                                        # Extract page/group URL
                                        if '/groups/' in href or '/pages/' in href or '/profile.php' in href:
                                            source_url = href.split('?')[0]  # Remove query params
                                            break
                            except:
                                pass
                            
                            # Process the incident
                            incident = self.nlp.process(text, 'Facebook', source_url or search_url)
                            
                            if incident:
                                incidents.append(incident)
                                self.scraped_ids.add(text_id)
                                logger.info(f"  [FB] Found incident in {incident.area}")
                                
                                # Track discovered source
                                if source_url and source_url != search_url:
                                    if source_url not in discovered_sources:
                                        discovered_sources[source_url] = {
                                            'incidents': [],
                                            'type': 'group' if '/groups/' in source_url else 'page'
                                        }
                                    discovered_sources[source_url]['incidents'].append(incident)
                        
                        except Exception as e:
                            logger.debug(f"  [FB] Error processing post: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"  [FB] Error with selector {selector}: {e}")
                    continue
            
            # Step 2: Save discovered sources to knowledge base
            if discovered_sources and hasattr(self, 'knowledge_base'):
                for source_url, data in discovered_sources.items():
                    incident_count = len(data['incidents'])
                    quality_avg = sum(inc.quality_score for inc in data['incidents']) / incident_count
                    
                    self.knowledge_base.discover_facebook_source(
                        url=source_url,
                        source_type=data['type'],
                        incidents_found=incident_count,
                        quality_avg=quality_avg
                    )
                    logger.info(f"  [FB-DISCOVERY] Saved {data['type']}: {source_url} ({incident_count} incidents, quality: {quality_avg:.2f})")
            
            # Step 3: Scrape previously discovered high-quality sources
            if hasattr(self, 'knowledge_base'):
                discovered = self.knowledge_base.get_discovered_facebook_sources(min_quality=0.7, min_incidents=2)
                logger.info(f"  [FB] Scraping {len(discovered)} previously discovered sources")
                
                for source in discovered[:5]:  # Limit to top 5 sources
                    try:
                        logger.info(f"  [FB] Scraping discovered {source['type']}: {source['url']}")
                        self.driver.get(source['url'])
                        time.sleep(4)
                        
                        # Scroll to load posts
                        for _ in range(2):
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(2)
                        
                        # Find posts
                        posts = self.driver.find_elements(By.CSS_SELECTOR, 'div[role="article"]')
                        
                        for post in posts[:10]:
                            try:
                                text = post.text
                                if len(text) < 25:
                                    continue
                                
                                text_id = hashlib.md5(text.encode()).hexdigest()
                                if text_id in self.scraped_ids:
                                    continue
                                
                                incident = self.nlp.process(text, f"Facebook ({source['type']})", source['url'])
                                
                                if incident:
                                    incidents.append(incident)
                                    self.scraped_ids.add(text_id)
                                    logger.info(f"  [FB-DISCOVERED] Found in {incident.area}")
                            except:
                                continue
                    
                    except Exception as e:
                        logger.warning(f"  [FB] Error scraping discovered source: {e}")
                        # Mark as inactive if consistently failing
                        continue
            
            logger.info(f"  [FB] Total incidents found: {len(incidents)}")
            logger.info(f"  [FB] New sources discovered: {len(discovered_sources)}")
            
        except Exception as e:
            logger.error(f"  [FB] Error: {e}")
        
        return incidents

    def scrape_twitter_deep(self, query: str) -> List[CrimeIncident]:
        """
        Scrape Twitter with API first, fallback to Selenium
        Uses official Twitter API v2 with proper rate limiting
        """
        incidents = []
        
        # Try Twitter API first (official, respects rate limits)
        if self.twitter_scraper.enabled and not self.twitter_scraper.rate_limit_reached:
            logger.info(f"  [Twitter] Using Twitter API for query: {query}")
            tweets = self.twitter_scraper.search_tweets(query, max_results=20)
            
            for tweet_data in tweets:
                try:
                    text = tweet_data.get('text', '')
                    url = tweet_data.get('url', '')
                    timestamp = tweet_data.get('timestamp', '')
                    source = tweet_data.get('source', 'Twitter')
                    
                    if len(text) < 25:
                        continue
                    
                    text_id = hashlib.md5(text.encode()).hexdigest()
                    if text_id in self.scraped_ids:
                        continue
                    
                    incident = self.nlp.process_text(text, source, url, timestamp)
                    
                    if incident:
                        incidents.append(incident)
                        self.scraped_ids.add(text_id)
                        logger.info(f"  [Twitter-API] Found incident: {incident.area} - {incident.incident_type}")
                
                except Exception as e:
                    logger.debug(f"  [Twitter-API] Error processing tweet: {e}")
                    continue
            
            if incidents:
                logger.info(f"  [Twitter-API] Found {len(incidents)} incidents via Twitter API")
                return incidents
            elif self.twitter_scraper.rate_limit_reached:
                logger.warning(f"  [Twitter-API] Rate limit reached, skipping Selenium fallback")
                return incidents
            else:
                logger.info(f"  [Twitter-API] No incidents found, falling back to Selenium")
        
        # Fallback to Selenium scraping if API didn't work or is rate limited
        # Note: Selenium scraping of Twitter often requires login, so this may not work
        try:
            self.init_driver()
            url = f"https://twitter.com/search?q={query.replace(' ', '%20')}&f=live"
            
            self.driver.get(url)
            time.sleep(5)
            
            # Scroll multiple times to load more tweets
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            tweets = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
            
            for tweet in tweets[:20]:
                try:
                    text = tweet.text
                    
                    if len(text) < 25:
                        continue
                    
                    text_id = hashlib.md5(text.encode()).hexdigest()
                    if text_id in self.scraped_ids:
                        continue
                    
                    incident = self.nlp.process_text(text, 'Twitter', url)
                    
                    if incident:
                        incidents.append(incident)
                        self.scraped_ids.add(text_id)
                
                except:
                    continue
        
        except Exception as e:
            logger.error(f"Twitter Selenium error: {e}")
        
        return incidents
    
    def scrape_rss_feeds(self, query: str) -> List[CrimeIncident]:
        """Scrape RSS feeds from Pakistani news sites"""
        incidents = []
        
        # Pakistani news RSS feeds
        rss_feeds = [
            'https://www.dawn.com/feeds/home',
            'https://www.dawn.com/feeds/karachi',
            'https://www.geo.tv/rss/1/1',
            'https://tribune.com.pk/feed/home',
            'https://www.express.pk/feed/',
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:
                    try:
                        # Combine title and summary
                        text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                        
                        if len(text) < 25:
                            continue
                        
                        text_id = hashlib.md5(text.encode()).hexdigest()
                        if text_id in self.scraped_ids:
                            continue
                        
                        incident = self.nlp.process(text, f'RSS: {feed_url.split("/")[2]}', entry.get('link', feed_url))
                        
                        if incident:
                            incidents.append(incident)
                            self.scraped_ids.add(text_id)
                    
                    except:
                        continue
            
            except Exception as e:
                logger.debug(f"RSS feed error {feed_url}: {e}")
                continue
        
        return incidents
    
    def scrape_news_site_deep(self, site: str, query: str) -> List[CrimeIncident]:
        """Scrape Pakistani news sites with deep crawling"""
        incidents = []
        max_retries = 2
        
        urls = {
            'dawn': f"https://www.dawn.com/search?q={query.replace(' ', '+')}",
            'geo': f"https://www.geo.tv/search/{query.replace(' ', '-')}",
            'tribune': f"https://tribune.com.pk/search?q={query.replace(' ', '+')}",
            'express': f"https://www.express.pk/search/?q={query.replace(' ', '+')}"
        }
        
        if site not in urls:
            return incidents
        
        for attempt in range(max_retries):
            try:
                self.init_driver()
                url = urls[site]
                
                self.driver.get(url)
                time.sleep(2)
                
                # Scroll to load more
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Find article links
                article_links = []
                link_elements = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/story/"], a[href*="/news/"], a[href*="/article/"]')
                
                for link in link_elements[:10]:
                    try:
                        href = link.get_attribute('href')
                        if href and href not in article_links:
                            article_links.append(href)
                    except:
                        continue
                
                # Visit each article for deep content
                for article_url in article_links[:5]:
                    try:
                        self.driver.get(article_url)
                        time.sleep(1)
                        
                        # Get article content
                        content_selectors = [
                            'article',
                            'div.story-content',
                            'div.article-content',
                            'div.post-content',
                            'div.entry-content',
                            'p'
                        ]
                        
                        for selector in content_selectors:
                            try:
                                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                
                                for elem in elements[:5]:
                                    try:
                                        text = elem.text
                                        
                                        if len(text) < 25:
                                            continue
                                        
                                        text_id = hashlib.md5(text.encode()).hexdigest()
                                        if text_id in self.scraped_ids:
                                            continue
                                        
                                        incident = self.nlp.process(text, site.title(), article_url)
                                        
                                        if incident:
                                            incidents.append(incident)
                                            self.scraped_ids.add(text_id)
                                    
                                    except:
                                        continue
                            except:
                                continue
                    
                    except:
                        continue
                
                break  # Success, exit retry loop
            
            except WebDriverException as e:
                logger.error(f"{site} driver error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self.restart_driver()
                else:
                    logger.error(f"{site}: Max retries reached")
            
            except Exception as e:
                logger.error(f"{site} error: {e}")
                break
        
        return incidents


# ============================================================================
# TASK 19: SCRAPER ORCHESTRATOR AS MAIN COORDINATOR
# ============================================================================

class ExecutionMode(Enum):
    """Execution modes for the scraper orchestrator"""
    FULL = "full"              # Comprehensive scraping (all sources, all queries)
    QUICK = "quick"            # High-priority sources only (fast execution)
    SCHEDULED = "scheduled"    # Optimized for cron/scheduled runs (< 30 minutes)
    TEST = "test"              # Validation mode (test setup and connectivity)
    DIAGNOSTIC = "diagnostic"  # Troubleshooting mode (health checks and diagnostics)


class ScraperOrchestrator:
    """
    Central coordinator for all scraping operations with comprehensive health monitoring,
    self-diagnostics, and graceful error handling.
    
    Features:
    - Health checks for all components before starting
    - Self-diagnostic mode for troubleshooting
    - Multiple execution modes (full, quick, scheduled, test, diagnostic)
    - Progress reporting and statistics generation
    - Graceful shutdown with data saving
    - Component coordination (drivers, APIs, storage)
    - Execution monitoring and error recovery
    
    Requirements: 23, 25, 26, 29
    """
    
    def __init__(
        self,
        config: Configuration,
        mode: ExecutionMode = ExecutionMode.FULL,
        target: int = 100
    ):
        """
        Initialize scraper orchestrator
        
        Args:
            config: Configuration object with API keys and settings
            mode: Execution mode (default: FULL)
            target: Target number of incidents to collect (default: 100)
        """
        self.config = config
        self.mode = mode
        self.target = target
        
        # Execution state
        self.start_time = None
        self.end_time = None
        self.is_running = False
        self.should_stop = False
        
        # Components (initialized in health_check)
        self.driver_pool: Optional[DriverPoolManager] = None
        self.http_client: Optional[HTTPClientPool] = None
        self.error_recovery: Optional[ErrorRecoverySystem] = None
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.incident_store: Optional[IncidentStore] = None
        self.deduplicator: Optional[DeduplicationSystem] = None
        
        # Statistics
        self.stats = {
            'incidents_collected': 0,
            'queries_executed': 0,
            'sources_scraped': 0,
            'errors_encountered': 0,
            'duplicates_filtered': 0,
            'execution_time_seconds': 0,
            'incidents_per_minute': 0.0,
            'avg_quality_score': 0.0,
            'avg_confidence_score': 0.0
        }
        
        # Register shutdown handler
        atexit.register(self._emergency_shutdown)
        
        logging.info("="*80)
        logging.info(f"[ScraperOrchestrator] Initialized in {mode.value.upper()} mode")
        logging.info(f"[ScraperOrchestrator] Target: {target} incidents")
        logging.info("="*80)
    
    def health_check(self) -> Dict[str, bool]:
        """
        Validate all components before starting scraping.
        Checks driver pool, HTTP client, API connectivity, file access, etc.
        
        Returns:
            Dictionary with component health status
            
        Requirements: 26
        """
        logging.info("\n" + "="*80)
        logging.info("[HEALTH CHECK] Validating all components...")
        logging.info("="*80)
        
        health_status = {
            'configuration': False,
            'driver_pool': False,
            'http_client': False,
            'openrouter_api': False,
            'reddit_api': False,
            'file_system': False,
            'knowledge_base': False,
            'overall': False
        }
        
        # 1. Configuration check
        try:
            logging.info("[Health] Checking configuration...")
            if self.config:
                health_status['configuration'] = True
                logging.info("  ✓ Configuration loaded")
                
                # Check API keys
                if self.config.has_openrouter_api():
                    logging.info("  ✓ OpenRouter API key configured")
                else:
                    logging.warning("  ⚠ OpenRouter API key not configured (LLM processing disabled)")
                
                if self.config.has_reddit_api():
                    logging.info("  ✓ Reddit API credentials configured")
                else:
                    logging.warning("  ⚠ Reddit API credentials not configured (Reddit scraping disabled)")
            else:
                logging.error("  ✗ Configuration not loaded")
        except Exception as e:
            logging.error(f"  ✗ Configuration check failed: {e}")
        
        # 2. Driver pool check
        try:
            logging.info("[Health] Checking driver pool...")
            self.driver_pool = DriverPoolManager(pool_size=3)
            
            # Try to create a test driver
            test_driver = self.driver_pool.get_driver()
            if test_driver and self.driver_pool.check_health(test_driver):
                health_status['driver_pool'] = True
                logging.info("  ✓ Driver pool operational")
                self.driver_pool.release_driver(test_driver)
            else:
                logging.error("  ✗ Driver pool health check failed")
        except Exception as e:
            logging.error(f"  ✗ Driver pool initialization failed: {e}")
        
        # 3. HTTP client check
        try:
            logging.info("[Health] Checking HTTP client pool...")
            self.http_client = HTTPClientPool(pool_size=10, max_retries=3, timeout=30)
            
            # Test HTTP connectivity
            test_response = self.http_client.get('https://www.google.com', timeout=10)
            if test_response and test_response.status_code == 200:
                health_status['http_client'] = True
                logging.info("  ✓ HTTP client pool operational")
            else:
                logging.warning("  ⚠ HTTP client test request failed")
        except Exception as e:
            logging.error(f"  ✗ HTTP client initialization failed: {e}")
        
        # 4. OpenRouter API check
        if self.config.has_openrouter_api():
            try:
                logging.info("[Health] Checking OpenRouter API connectivity...")
                # Simple connectivity test (don't make actual API call to save costs)
                health_status['openrouter_api'] = True
                logging.info("  ✓ OpenRouter API configured (connectivity not tested)")
            except Exception as e:
                logging.error(f"  ✗ OpenRouter API check failed: {e}")
        else:
            health_status['openrouter_api'] = True  # Not required
            logging.info("  ⊘ OpenRouter API not configured (skipped)")
        
        # 5. Reddit API check
        if self.config.has_reddit_api() and HAS_PRAW:
            try:
                logging.info("[Health] Checking Reddit API connectivity...")
                # Test PRAW initialization
                reddit = praw.Reddit(
                    client_id=self.config.REDDIT_CLIENT_ID,
                    client_secret=self.config.REDDIT_CLIENT_SECRET,
                    user_agent=self.config.REDDIT_USER_AGENT
                )
                # Test read-only access
                reddit.read_only = True
                health_status['reddit_api'] = True
                logging.info("  ✓ Reddit API operational")
            except Exception as e:
                logging.error(f"  ✗ Reddit API check failed: {e}")
        else:
            health_status['reddit_api'] = True  # Not required
            logging.info("  ⊘ Reddit API not configured or PRAW not installed (skipped)")
        
        # 6. File system check
        try:
            logging.info("[Health] Checking file system access...")
            
            # Check output directory
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # Check data directory
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            
            # Check logs directory
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)
            
            # Test write access
            test_file = output_dir / '.health_check_test'
            test_file.write_text('test')
            test_file.unlink()
            
            health_status['file_system'] = True
            logging.info("  ✓ File system access operational")
        except Exception as e:
            logging.error(f"  ✗ File system check failed: {e}")
        
        # 7. Knowledge base check
        try:
            logging.info("[Health] Checking knowledge base...")
            self.knowledge_base = KnowledgeBase(knowledge_file='ultimate_knowledge.json')
            health_status['knowledge_base'] = True
            logging.info(f"  ✓ Knowledge base loaded (Run #{self.knowledge_base.knowledge.runs})")
        except Exception as e:
            logging.error(f"  ✗ Knowledge base check failed: {e}")
        
        # 8. Initialize other components
        try:
            logging.info("[Health] Initializing remaining components...")
            
            # Error recovery system
            self.error_recovery = ErrorRecoverySystem(driver_pool=self.driver_pool)
            logging.info("  ✓ Error recovery system initialized")
            
            # Progress tracker
            self.progress_tracker = ProgressTracker(progress_file='data/progress.json')
            logging.info("  ✓ Progress tracker initialized")
            
            # Incident store
            self.incident_store = IncidentStore(output_dir='output')
            logging.info("  ✓ Incident store initialized")
            
            # Deduplication system
            self.deduplicator = DeduplicationSystem(similarity_threshold=0.85, temporal_window_hours=48)
            
            # Load existing incidents for idempotency
            existing_incidents = self.incident_store.load_existing_incidents()
            for incident in existing_incidents:
                self.deduplicator.seen_ids.add(incident.incident_id)
            
            # Add recent incidents for similarity comparison
            cutoff_time = datetime.now() - timedelta(hours=48)
            for incident in existing_incidents:
                try:
                    if 'T' in incident.incident_date:
                        inc_time = datetime.fromisoformat(incident.incident_date)
                    else:
                        inc_time = datetime.strptime(incident.incident_date, '%Y-%m-%d')
                    
                    if inc_time >= cutoff_time:
                        self.deduplicator.recent_incidents.append(incident)
                except:
                    continue
            
            logging.info(f"  ✓ Deduplication system initialized ({len(self.deduplicator.seen_ids)} existing IDs)")
            
        except Exception as e:
            logging.error(f"  ✗ Component initialization failed: {e}")
        
        # Overall health status
        critical_components = ['configuration', 'driver_pool', 'http_client', 'file_system', 'knowledge_base']
        health_status['overall'] = all(health_status.get(comp, False) for comp in critical_components)
        
        # Summary
        logging.info("\n" + "="*80)
        if health_status['overall']:
            logging.info("[HEALTH CHECK] ✓ All critical components operational")
        else:
            logging.warning("[HEALTH CHECK] ⚠ Some components failed health check")
            failed = [k for k, v in health_status.items() if not v and k != 'overall']
            logging.warning(f"[HEALTH CHECK] Failed components: {', '.join(failed)}")
        logging.info("="*80 + "\n")
        
        return health_status
    
    def self_diagnostic(self) -> Dict:
        """
        Perform comprehensive self-diagnostic to check for common issues.
        Useful for troubleshooting before starting scraping.
        
        Returns:
            Dictionary with diagnostic results
            
        Requirements: 26
        """
        logging.info("\n" + "="*80)
        logging.info("[SELF-DIAGNOSTIC] Running comprehensive diagnostics...")
        logging.info("="*80)
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'mode': self.mode.value,
            'issues_found': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 1. Check environment variables
        logging.info("[Diagnostic] Checking environment variables...")
        if not self.config.OPENROUTER_API_KEY:
            diagnostics['warnings'].append("OpenRouter API key not configured - LLM processing will be disabled")
            diagnostics['recommendations'].append("Add OPENROUTER_API_KEY to .env file for Urdu translation and data extraction")
        
        if not self.config.REDDIT_CLIENT_ID or not self.config.REDDIT_CLIENT_SECRET:
            diagnostics['warnings'].append("Reddit API credentials not configured - Reddit scraping will be disabled")
            diagnostics['recommendations'].append("Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env file for Reddit scraping")
        
        # 2. Check disk space
        logging.info("[Diagnostic] Checking disk space...")
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free / (1024**3)
            
            if free_gb < 1.0:
                diagnostics['issues_found'].append(f"Low disk space: {free_gb:.2f} GB free")
                diagnostics['recommendations'].append("Free up disk space before running scraper")
            elif free_gb < 5.0:
                diagnostics['warnings'].append(f"Disk space getting low: {free_gb:.2f} GB free")
            
            logging.info(f"  Disk space: {free_gb:.2f} GB free")
        except Exception as e:
            logging.warning(f"  Could not check disk space: {e}")
        
        # 3. Check memory usage
        logging.info("[Diagnostic] Checking memory usage...")
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 0.5:
                diagnostics['issues_found'].append(f"Low memory: {available_gb:.2f} GB available")
                diagnostics['recommendations'].append("Close other applications to free up memory")
            elif available_gb < 2.0:
                diagnostics['warnings'].append(f"Memory getting low: {available_gb:.2f} GB available")
            
            logging.info(f"  Available memory: {available_gb:.2f} GB")
        except ImportError:
            logging.warning("  psutil not installed - cannot check memory usage")
        except Exception as e:
            logging.warning(f"  Could not check memory: {e}")
        
        # 4. Check Chrome/ChromeDriver
        logging.info("[Diagnostic] Checking Chrome/ChromeDriver...")
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            driver_path = ChromeDriverManager().install()
            logging.info(f"  ✓ ChromeDriver available: {driver_path}")
        except Exception as e:
            diagnostics['issues_found'].append(f"ChromeDriver check failed: {e}")
            diagnostics['recommendations'].append("Install Chrome browser and ensure webdriver-manager can download ChromeDriver")
        
        # 5. Check network connectivity
        logging.info("[Diagnostic] Checking network connectivity...")
        test_urls = [
            ('Google', 'https://www.google.com'),
            ('Facebook', 'https://www.facebook.com'),
            ('Twitter', 'https://www.twitter.com'),
            ('Dawn News', 'https://www.dawn.com')
        ]
        
        for name, url in test_urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logging.info(f"  ✓ {name} accessible")
                else:
                    logging.warning(f"  ⚠ {name} returned status {response.status_code}")
            except Exception as e:
                diagnostics['warnings'].append(f"Cannot access {name}: {e}")
                logging.warning(f"  ✗ {name} not accessible: {e}")
        
        # 6. Check existing data
        logging.info("[Diagnostic] Checking existing data...")
        try:
            output_dir = Path('output')
            excel_file = output_dir / 'karachi_crimes.xlsx'
            
            if excel_file.exists():
                size_mb = excel_file.stat().st_size / (1024**2)
                logging.info(f"  Existing data file: {excel_file} ({size_mb:.2f} MB)")
                
                # Check if file is too large
                if size_mb > 100:
                    diagnostics['warnings'].append(f"Output file is large ({size_mb:.2f} MB) - consider archiving old data")
            else:
                logging.info("  No existing data file - starting fresh")
        except Exception as e:
            logging.warning(f"  Could not check existing data: {e}")
        
        # 7. Check log files
        logging.info("[Diagnostic] Checking log files...")
        try:
            logs_dir = Path('logs')
            if logs_dir.exists():
                log_files = list(logs_dir.glob('*.log'))
                total_size = sum(f.stat().st_size for f in log_files) / (1024**2)
                logging.info(f"  Log files: {len(log_files)} files, {total_size:.2f} MB total")
                
                if total_size > 100:
                    diagnostics['warnings'].append(f"Log files are large ({total_size:.2f} MB) - consider rotating or archiving")
        except Exception as e:
            logging.warning(f"  Could not check log files: {e}")
        
        # 8. Check for stale lock files
        logging.info("[Diagnostic] Checking for stale lock files...")
        lock_file = Path('.scraper.lock')
        if lock_file.exists():
            try:
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age > 7200:  # 2 hours
                    diagnostics['warnings'].append(f"Stale lock file found (age: {lock_age/3600:.1f} hours)")
                    diagnostics['recommendations'].append("Remove .scraper.lock file if no other instance is running")
                else:
                    diagnostics['issues_found'].append("Lock file exists - another instance may be running")
            except Exception as e:
                logging.warning(f"  Could not check lock file: {e}")
        
        # Summary
        logging.info("\n" + "="*80)
        logging.info("[SELF-DIAGNOSTIC] Summary:")
        logging.info(f"  Issues found: {len(diagnostics['issues_found'])}")
        logging.info(f"  Warnings: {len(diagnostics['warnings'])}")
        logging.info(f"  Recommendations: {len(diagnostics['recommendations'])}")
        
        if diagnostics['issues_found']:
            logging.warning("\n  Critical Issues:")
            for issue in diagnostics['issues_found']:
                logging.warning(f"    - {issue}")
        
        if diagnostics['warnings']:
            logging.info("\n  Warnings:")
            for warning in diagnostics['warnings']:
                logging.info(f"    - {warning}")
        
        if diagnostics['recommendations']:
            logging.info("\n  Recommendations:")
            for rec in diagnostics['recommendations']:
                logging.info(f"    - {rec}")
        
        logging.info("="*80 + "\n")
        
        return diagnostics
    
    def run(self) -> List[CrimeIncident]:
        """
        Main execution method that coordinates all scraping engines.
        Handles different execution modes and orchestrates the entire scraping process.
        
        Returns:
            List of collected CrimeIncident objects
            
        Requirements: 23, 25, 26, 29
        """
        self.start_time = datetime.now()
        self.is_running = True
        
        try:
            # Run health check
            health_status = self.health_check()
            
            if not health_status['overall']:
                logging.error("[ScraperOrchestrator] Health check failed - cannot proceed")
                if self.mode == ExecutionMode.DIAGNOSTIC:
                    # In diagnostic mode, run diagnostics even if health check fails
                    self.self_diagnostic()
                return []
            
            # Handle different execution modes
            if self.mode == ExecutionMode.DIAGNOSTIC:
                # Run diagnostics and exit
                self.self_diagnostic()
                logging.info("[ScraperOrchestrator] Diagnostic mode complete")
                return []
            
            elif self.mode == ExecutionMode.TEST:
                # Test mode - validate setup and exit
                logging.info("\n" + "="*80)
                logging.info("[TEST MODE] All systems operational - ready for scraping")
                logging.info("="*80 + "\n")
                return []
            
            # For FULL, QUICK, and SCHEDULED modes, proceed with scraping
            logging.info("\n" + "="*80)
            logging.info(f"[ScraperOrchestrator] Starting scraping in {self.mode.value.upper()} mode")
            logging.info("="*80 + "\n")
            
            # Use existing UltimateOrchestrator for actual scraping
            # (This maintains compatibility with existing implementation)
            orchestrator = UltimateOrchestrator(self.config, target=self.target)
            incidents = orchestrator.run()
            
            # Update statistics
            self.stats['incidents_collected'] = len(incidents)
            self.end_time = datetime.now()
            self._calculate_statistics(incidents)
            
            # Generate progress report
            self._generate_progress_report(incidents)
            
            return incidents
            
        except KeyboardInterrupt:
            logging.info("\n[ScraperOrchestrator] Execution interrupted by user")
            self.should_stop = True
            raise
        
        except Exception as e:
            logging.error(f"[ScraperOrchestrator] Fatal error during execution: {e}")
            logging.error(traceback.format_exc())
            self.stats['errors_encountered'] += 1
            raise
        
        finally:
            self.is_running = False
            if self.end_time is None:
                self.end_time = datetime.now()
            
            # Calculate execution time
            if self.start_time and self.end_time:
                self.stats['execution_time_seconds'] = (self.end_time - self.start_time).total_seconds()
    
    def _calculate_statistics(self, incidents: List[CrimeIncident]):
        """Calculate statistics from collected incidents"""
        if not incidents:
            return
        
        # Calculate averages
        total_quality = sum(inc.quality_score for inc in incidents)
        total_confidence = sum(inc.confidence_score for inc in incidents)
        
        self.stats['avg_quality_score'] = total_quality / len(incidents)
        self.stats['avg_confidence_score'] = total_confidence / len(incidents)
        
        # Calculate incidents per minute
        if self.stats['execution_time_seconds'] > 0:
            self.stats['incidents_per_minute'] = (
                len(incidents) / (self.stats['execution_time_seconds'] / 60)
            )
    
    def _generate_progress_report(self, incidents: List[CrimeIncident]):
        """
        Generate comprehensive progress report with statistics.
        
        Requirements: 26
        """
        logging.info("\n" + "="*80)
        logging.info("[PROGRESS REPORT]")
        logging.info("="*80)
        
        # Execution summary
        logging.info(f"\nExecution Mode: {self.mode.value.upper()}")
        logging.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}")
        logging.info(f"End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}")
        
        if self.stats['execution_time_seconds'] > 0:
            logging.info(f"Duration: {self.stats['execution_time_seconds']:.1f} seconds ({self.stats['execution_time_seconds']/60:.1f} minutes)")
        else:
            logging.info("Duration: N/A")
        
        # Collection statistics
        logging.info(f"\nIncidents Collected: {self.stats['incidents_collected']}")
        logging.info(f"Target: {self.target}")
        logging.info(f"Completion: {(self.stats['incidents_collected']/self.target*100):.1f}%")
        
        # Performance metrics
        logging.info(f"\nPerformance:")
        logging.info(f"  Incidents/minute: {self.stats['incidents_per_minute']:.2f}")
        logging.info(f"  Avg quality score: {self.stats['avg_quality_score']:.3f}")
        logging.info(f"  Avg confidence score: {self.stats['avg_confidence_score']:.3f}")
        
        # Component statistics
        if self.driver_pool:
            pool_stats = self.driver_pool.get_pool_stats()
            logging.info(f"\nDriver Pool:")
            logging.info(f"  Active drivers: {pool_stats['pool_size']}/{pool_stats['max_pool_size']}")
            logging.info(f"  Total drivers created: {pool_stats['total_drivers_created']}")
        
        if self.error_recovery:
            error_stats = self.error_recovery.get_error_statistics()
            logging.info(f"\nError Recovery:")
            logging.info(f"  Total errors: {error_stats['total_errors']}")
            if error_stats['error_patterns']:
                logging.info(f"  Error patterns:")
                for error_type, count in error_stats['error_patterns'].items():
                    logging.info(f"    - {error_type}: {count}")
        
        if self.deduplicator:
            dedup_stats = self.deduplicator.get_stats()
            logging.info(f"\nDeduplication:")
            logging.info(f"  Total unique IDs: {dedup_stats['total_ids']}")
            logging.info(f"  Recent incidents tracked: {dedup_stats['recent_incidents']}")
        
        logging.info("\n" + "="*80 + "\n")
    
    def graceful_shutdown(self, incidents: List[CrimeIncident] = None):
        """
        Perform graceful shutdown with data saving.
        Ensures all data is saved before exiting.
        
        Args:
            incidents: Optional list of incidents to save before shutdown
            
        Requirements: 26
        """
        logging.info("\n" + "="*80)
        logging.info("[GRACEFUL SHUTDOWN] Shutting down gracefully...")
        logging.info("="*80)
        
        try:
            # Save incidents if provided
            if incidents and self.incident_store:
                logging.info("[Shutdown] Saving collected incidents...")
                success = self.incident_store.save_incidents(incidents, append_mode=True)
                if success:
                    logging.info(f"[Shutdown] ✓ Saved {len(incidents)} incidents")
                else:
                    logging.error("[Shutdown] ✗ Failed to save incidents")
            
            # Save progress
            if self.progress_tracker:
                logging.info("[Shutdown] Saving progress checkpoint...")
                try:
                    self.progress_tracker.save_checkpoint()
                    logging.info("[Shutdown] ✓ Progress saved")
                except Exception as e:
                    logging.error(f"[Shutdown] ✗ Failed to save progress: {e}")
            
            # Save knowledge base
            if self.knowledge_base:
                logging.info("[Shutdown] Saving knowledge base...")
                try:
                    self.knowledge_base.save()
                    logging.info("[Shutdown] ✓ Knowledge base saved")
                except Exception as e:
                    logging.error(f"[Shutdown] ✗ Failed to save knowledge base: {e}")
            
            # Close driver pool
            if self.driver_pool:
                logging.info("[Shutdown] Closing driver pool...")
                try:
                    self.driver_pool.close_all()
                    logging.info("[Shutdown] ✓ Driver pool closed")
                except Exception as e:
                    logging.error(f"[Shutdown] ✗ Failed to close driver pool: {e}")
            
            # Close HTTP sessions
            if self.http_client:
                logging.info("[Shutdown] Closing HTTP sessions...")
                try:
                    self.http_client.close_all()
                    logging.info("[Shutdown] ✓ HTTP sessions closed")
                except Exception as e:
                    logging.error(f"[Shutdown] ✗ Failed to close HTTP sessions: {e}")
            
            logging.info("[Shutdown] Graceful shutdown complete")
            
        except Exception as e:
            logging.error(f"[Shutdown] Error during graceful shutdown: {e}")
            logging.error(traceback.format_exc())
        
        finally:
            self.is_running = False
            logging.info("="*80 + "\n")
    
    def _emergency_shutdown(self):
        """Emergency shutdown handler (called by atexit)"""
        if self.is_running:
            logging.warning("[EMERGENCY SHUTDOWN] Unexpected exit detected")
            self.graceful_shutdown()
    
    def get_statistics(self) -> Dict:
        """
        Get current execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return dict(self.stats)


class UltimateOrchestrator:
    """Ultimate orchestrator with all sources"""
    
    def __init__(self, config: Configuration, target: int = 100):
        self.config = config
        self.target = target
        self.learning = AILearning()
        self.scraper = UltimateScraper(config, knowledge_base=self.learning)
        self.all_incidents = []
        self.queries_used = []
        
        # Initialize deduplication system with similarity threshold 0.85 and 48-hour window
        self.deduplicator = DeduplicationSystem(similarity_threshold=0.85, temporal_window_hours=48)
        
        # Initialize incident store for idempotent operations
        self.incident_store = IncidentStore(output_dir='output')
        
        # Initialize geocoding service
        self.geocoding_service = None
        if config.has_google_maps_api():
            self.geocoding_service = GeocodingService(
                api_key=config.GOOGLE_MAPS_API_KEY,
                http_client=self.scraper.http_client if hasattr(self.scraper, 'http_client') else None
            )
            # Load geocoding cache
            self.geocoding_service.load_cache()
            logger.info("[Geocoding] Service initialized and cache loaded")
        else:
            logger.warning("[Geocoding] Service disabled - no API key configured")
        
        # Load existing incident IDs for idempotency
        self._load_existing_incidents()
        
        logger.info("="*80)
        logger.info("ULTIMATE KARACHI CRIME SCRAPER - FINAL VERSION")
        logger.info(f"Run #{self.learning.knowledge.runs + 1}")
        logger.info(f"Target: {target} incidents")
        logger.info("="*80)
    
    def _geocode_incidents(self, incidents: List[CrimeIncident]):
        """
        Geocode a list of incidents using Google Maps API
        
        Args:
            incidents: List of CrimeIncident objects to geocode
        """
        if not self.geocoding_service:
            return
        
        geocoded_count = 0
        for incident in incidents:
            # Skip if already geocoded
            if incident.latitude is not None and incident.longitude is not None:
                continue
            
            # Geocode the location
            coords = self.geocoding_service.geocode_location(
                area=incident.area,
                sub_area=incident.sub_area,
                city=incident.city
            )
            
            # Update incident with coordinates
            incident.latitude = coords.get('latitude')
            incident.longitude = coords.get('longitude')
            
            if incident.latitude is not None:
                geocoded_count += 1
        
        if geocoded_count > 0:
            logger.info(f"  [Geocoding] Geocoded {geocoded_count}/{len(incidents)} incidents")
    
    def _load_existing_incidents(self):
        """
        Load existing incidents from output directory for idempotency.
        Uses IncidentStore to load existing data and populate deduplication system.
        
        Requirements: 2, 22
        """
        # Load existing incidents using IncidentStore
        existing_incidents = self.incident_store.load_existing_incidents()
        
        if not existing_incidents:
            logger.info("[Deduplication] No existing incidents found - starting fresh")
            return
        
        # Populate deduplication system with existing IDs
        for incident in existing_incidents:
            self.deduplicator.seen_ids.add(incident.incident_id)
        
        # Add recent incidents for similarity comparison (last 48 hours)
        cutoff_time = datetime.now() - timedelta(hours=48)
        for incident in existing_incidents:
            try:
                # Parse incident date
                if 'T' in incident.incident_date:
                    inc_time = datetime.fromisoformat(incident.incident_date)
                else:
                    inc_time = datetime.strptime(incident.incident_date, '%Y-%m-%d')
                
                # Only add if within temporal window
                if inc_time >= cutoff_time:
                    self.deduplicator.recent_incidents.append(incident)
            except Exception as e:
                logging.debug(f"[Deduplication] Error parsing incident date: {e}")
                continue
        
        # Log deduplication stats
        stats = self.deduplicator.get_stats()
        logger.info(f"[Deduplication] Loaded {stats['total_ids']} existing IDs")
        logger.info(f"[Deduplication] {stats['recent_incidents']} recent incidents for similarity comparison")
        logger.info("="*80)
    
    def run(self):
        try:
            queries = self.learning.get_queries(max_queries=30)
            
            for idx, query in enumerate(queries, 1):
                if len(self.all_incidents) >= self.target:
                    break
                
                logger.info(f"\n[{idx}/{len(queries)}] {query}")
                self.queries_used.append(query)
                
                # All sources for each query - ALL ENABLED
                sources = [
                    ('Google News', lambda q: self.scraper.scrape_google_news(q)),
                    ('RSS Feeds', lambda q: self.scraper.scrape_rss_feeds(q)),
                    ('Dawn', lambda q: self.scraper.scrape_news_site_deep('dawn', q)),
                    ('Geo', lambda q: self.scraper.scrape_news_site_deep('geo', q)),
                    ('Tribune', lambda q: self.scraper.scrape_news_site_deep('tribune', q)),
                    ('Express', lambda q: self.scraper.scrape_news_site_deep('express', q)),
                    # Facebook enabled with Graph API + auto-discovery:
                    ('Facebook Public', lambda q: self.scraper.scrape_facebook_public(q)),
                    # Twitter enabled with API v2:
                    ('Twitter', lambda q: self.scraper.scrape_twitter_deep(q)),
                ]
                
                query_incidents = []
                
                for source_name, scrape_func in sources:
                    if len(self.all_incidents) >= self.target:
                        break
                    
                    try:
                        new_incidents = scrape_func(query)
                        
                        if new_incidents:
                            # Apply deduplication to new incidents
                            unique_incidents = []
                            duplicate_count = 0
                            
                            for incident in new_incidents:
                                # Check if duplicate
                                is_dup, dup_id = self.deduplicator.is_duplicate(incident)
                                
                                if is_dup:
                                    duplicate_count += 1
                                    logger.debug(f"  [Dedup] Skipped duplicate from {source_name}: {incident.area}")
                                else:
                                    # Add to deduplication system
                                    self.deduplicator.add_incident(incident)
                                    unique_incidents.append(incident)
                            
                            if unique_incidents:
                                # Geocode incidents if service is available
                                if self.geocoding_service:
                                    self._geocode_incidents(unique_incidents)
                                
                                query_incidents.extend(unique_incidents)
                                logger.info(f"  {source_name}: +{len(unique_incidents)} ({duplicate_count} duplicates filtered)")
                                
                                # IMMEDIATE SAVE: Save each batch of incidents as soon as they're collected
                                logger.info(f"  [INSTANT-SAVE] Saving {len(unique_incidents)} new incidents immediately...")
                                try:
                                    # Add to all_incidents first so they're tracked
                                    temp_all = self.all_incidents + query_incidents
                                    
                                    # Debug: Check if incidents have IDs
                                    logger.debug(f"  [INSTANT-SAVE] Sample incident ID: {unique_incidents[0].incident_id if unique_incidents else 'N/A'}")
                                    
                                    success = self.incident_store.save_incidents(temp_all, append_mode=True)
                                    if success:
                                        logger.info(f"  [INSTANT-SAVE] ✓ Saved to {self.incident_store.excel_file}")
                                    else:
                                        logger.warning(f"  [INSTANT-SAVE] ✗ Failed to save incidents")
                                except Exception as e:
                                    logger.error(f"  [INSTANT-SAVE] Error: {e}")
                                    import traceback
                                    logger.error(f"  [INSTANT-SAVE] Traceback: {traceback.format_exc()}")
                            elif duplicate_count > 0:
                                logger.info(f"  {source_name}: 0 new ({duplicate_count} duplicates filtered)")
                    
                    except Exception as e:
                        logger.error(f"  {source_name} error: {e}")
                    
                    time.sleep(0.5)
                
                if query_incidents:
                    self.all_incidents.extend(query_incidents)
                    logger.info(f"  Total: +{len(query_incidents)} | Overall: {len(self.all_incidents)}/{self.target}")
                    
                    # IMMEDIATE SAVE: Save after each query completes
                    logger.info(f"[QUERY-SAVE] Saving all {len(self.all_incidents)} incidents after query...")
                    try:
                        success = self.incident_store.save_incidents(self.all_incidents, append_mode=True)
                        if success:
                            logger.info(f"[QUERY-SAVE] ✓ All incidents saved to disk")
                        else:
                            logger.warning(f"[QUERY-SAVE] ✗ Failed to save")
                    except Exception as e:
                        logger.error(f"[QUERY-SAVE] Error: {e}")
        
        finally:
            self.scraper.close_driver()
            self.learning.learn(self.all_incidents, self.queries_used)
            self.learning.save()
            
            # Save geocoding cache
            if self.geocoding_service:
                self.geocoding_service.save_cache()
                geo_stats = self.geocoding_service.get_statistics()
                logger.info(f"\n[GEOCODING STATS]")
                logger.info(f"  Total requests: {geo_stats['total_requests']}")
                logger.info(f"  Cache hits: {geo_stats['cache_hits']}")
                logger.info(f"  API calls: {geo_stats['api_calls']}")
                logger.info(f"  Successful: {geo_stats['successful_geocodes']}")
                logger.info(f"  Failed: {geo_stats['failed_geocodes']}")
                logger.info(f"  Fallback used: {geo_stats['fallback_used']}")
        
        # Log deduplication statistics
        dedup_stats = self.deduplicator.get_stats()
        logger.info(f"\n{'='*80}")
        logger.info(f"[DEDUPLICATION STATS]")
        logger.info(f"  Total unique incidents: {len(self.all_incidents)}")
        logger.info(f"  Total IDs tracked: {dedup_stats['total_ids']}")
        logger.info(f"  Recent incidents in memory: {dedup_stats['recent_incidents']}")
        logger.info(f"  Similarity cache size: {dedup_stats['cache_size']}")
        logger.info("="*80)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[DONE] {len(self.all_incidents)} unique incidents collected")
        logger.info("="*80)
        
        # Final save of all incidents
        if self.all_incidents:
            logger.info(f"\n[FINAL SAVE] Saving all {len(self.all_incidents)} incidents...")
            try:
                success = self.incident_store.save_incidents(self.all_incidents, append_mode=True)
                if success:
                    logger.info(f"[FINAL SAVE] ✓ Successfully saved all incidents")
                else:
                    logger.error("[FINAL SAVE] ✗ Failed to save incidents")
            except Exception as e:
                logger.error(f"[FINAL SAVE] Error: {e}")
        
        return self.all_incidents

    def save_results(self, incidents: List[CrimeIncident]):
        """
        Save results using IncidentStore with idempotent operations.
        Uses append-only mode to prevent duplicates across multiple runs.
        
        Requirements: 2, 22, 29
        """
        if not incidents:
            logger.warning("No incidents to save")
            return
        
        # Create incident store
        store = IncidentStore(output_dir='output')
        
        # Save incidents with append mode (idempotent)
        success = store.save_incidents(incidents, append_mode=True)
        
        if not success:
            logger.error("[ERROR] Failed to save incidents")
            return
        
        # Load all incidents (existing + new) for analytics
        all_incidents = store.load_existing_incidents()
        
        if not all_incidents:
            logger.warning("No incidents found for analytics")
            return
        
        # Create DataFrame for analytics
        df = pd.DataFrame([inc.to_dict() for inc in all_incidents])
        
        # Print analytics
        logger.info("\n" + "="*80)
        logger.info("[ANALYTICS]")
        logger.info("="*80)
        logger.info(f"\nTotal Incidents: {len(df)}")
        logger.info(f"New Incidents (this run): {len(incidents)}")
        logger.info(f"Avg quality: {df['quality_score'].mean():.3f}")
        logger.info(f"Avg confidence: {df['confidence_score'].mean():.3f}")
        logger.info(f"Unique areas: {df['area'].nunique()}")
        logger.info(f"With time: {(df['incident_time'] != 'Not specified').sum()}")
        logger.info(f"With device: {(df['device_model'] != '').sum()}")
        logger.info(f"Statistical incidents: {df['is_statistical'].sum()}")
        logger.info(f"Urdu translated: {df['is_urdu_translated'].sum()}")
        
        logger.info(f"\nTop 15 Areas:")
        for area, count in df['area'].value_counts().head(15).items():
            avg_q = df[df['area'] == area]['quality_score'].mean()
            logger.info(f"  {area}: {count} (Q:{avg_q:.2f})")
        
        logger.info(f"\nSources:")
        for source, count in df['source'].value_counts().items():
            avg_q = df[df['source'] == source]['quality_score'].mean()
            logger.info(f"  {source}: {count} (Q:{avg_q:.2f})")
        
        logger.info(f"\nIncident Types:")
        for itype, count in df['incident_type'].value_counts().items():
            logger.info(f"  {itype}: {count}")
        
        # Print store statistics
        store_stats = store.get_stats()
        logger.info(f"\n[STORAGE]")
        logger.info(f"Excel file: {store_stats['excel_file']}")
        logger.info(f"CSV backup: {store_stats['csv_file']}")
        logger.info(f"Backups: {store_stats['backup_count']}")
        if 'excel_size_mb' in store_stats:
            logger.info(f"Excel size: {store_stats['excel_size_mb']:.2f} MB")
        
        logger.info("\n" + "="*80)


def main():
    """Main entry point with execution lock and error handling"""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ultimate Karachi Crime Scraper')
    parser.add_argument('target', type=int, nargs='?', default=100,
                       help='Target number of incidents to collect (default: 100)')
    parser.add_argument('--mode', type=str, choices=['full', 'quick', 'scheduled', 'test', 'diagnostic'],
                       default='full', help='Execution mode (default: full)')
    parser.add_argument('--use-new-orchestrator', action='store_true',
                       help='Use new ScraperOrchestrator with health checks and diagnostics')
    
    args = parser.parse_args()
    target = args.target
    mode_str = args.mode
    use_new = args.use_new_orchestrator
    
    # Map mode string to ExecutionMode enum
    mode_map = {
        'full': ExecutionMode.FULL,
        'quick': ExecutionMode.QUICK,
        'scheduled': ExecutionMode.SCHEDULED,
        'test': ExecutionMode.TEST,
        'diagnostic': ExecutionMode.DIAGNOSTIC
    }
    mode = mode_map.get(mode_str, ExecutionMode.FULL)
    
    # Use execution lock to prevent concurrent runs
    lock = ExecutionLock()
    
    try:
        with lock:
            logger.info("="*80)
            logger.info("STARTING ULTIMATE KARACHI CRIME SCRAPER")
            logger.info(f"PID: {os.getpid()}")
            logger.info(f"Target: {target} incidents")
            logger.info(f"Mode: {mode_str.upper()}")
            if use_new:
                logger.info("Using: ScraperOrchestrator (with health checks)")
            else:
                logger.info("Using: UltimateOrchestrator (legacy mode)")
            logger.info("="*80)
            
            incidents = []
            orchestrator = None
            orch = None
            
            try:
                if use_new:
                    # Use new ScraperOrchestrator with health checks and diagnostics
                    orchestrator = ScraperOrchestrator(config, mode=mode, target=target)
                    
                    # Run scraping
                    incidents = orchestrator.run()
                    
                    # Save results immediately after collection (not in shutdown)
                    if mode not in [ExecutionMode.TEST, ExecutionMode.DIAGNOSTIC] and incidents:
                        logger.info(f"\n[SAVING] Saving {len(incidents)} collected incidents...")
                        success = orchestrator.incident_store.save_incidents(incidents, append_mode=True)
                        if success:
                            logger.info(f"[SAVING] ✓ Successfully saved {len(incidents)} incidents")
                        else:
                            logger.error("[SAVING] ✗ Failed to save incidents")
                else:
                    # Use legacy UltimateOrchestrator
                    orch = UltimateOrchestrator(config, target=target)
                    
                    # Run scraping
                    incidents = orch.run()
                    
                    # Save results immediately
                    if incidents:
                        logger.info(f"\n[SAVING] Saving {len(incidents)} collected incidents...")
                        orch.save_results(incidents)
            
            finally:
                # Always call graceful shutdown (without incidents, since already saved)
                if orchestrator:
                    orchestrator.graceful_shutdown()
                elif orch and hasattr(orch, 'scraper'):
                    orch.scraper.close_driver()
            
            logger.info("\n" + "="*80)
            logger.info("[SUCCESS] Scraping completed successfully!")
            logger.info("[TIP] Run again for even better results!")
            logger.info("="*80 + "\n")
    
    except KeyboardInterrupt:
        logger.info("\n\n" + "="*80)
        logger.info("[STOPPED] Execution interrupted by user")
        logger.info("="*80)
        
        # Try to save any collected incidents before exiting
        try:
            if 'incidents' in locals() and incidents:
                logger.info(f"[SAVING] Attempting to save {len(incidents)} collected incidents before exit...")
                if 'orchestrator' in locals() and orchestrator:
                    success = orchestrator.incident_store.save_incidents(incidents, append_mode=True)
                elif 'orch' in locals() and orch:
                    orch.save_results(incidents)
                    success = True
                else:
                    # Fallback: create new store and save
                    store = IncidentStore(output_dir='output')
                    success = store.save_incidents(incidents, append_mode=True)
                
                if success:
                    logger.info(f"[SAVING] ✓ Successfully saved {len(incidents)} incidents before exit")
                else:
                    logger.error("[SAVING] ✗ Failed to save incidents")
        except Exception as e:
            logger.error(f"[SAVING] Error saving incidents on interrupt: {e}")
        
        logger.info("="*80 + "\n")
        sys.exit(0)
    
    except RuntimeError as e:
        if "execution lock" in str(e).lower():
            logger.error(f"\n[ERROR] {e}")
            logger.error("Another instance is already running. Please wait for it to complete.")
            sys.exit(1)
        else:
            raise
    
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error(f"[ERROR] Fatal error occurred: {e}")
        logger.error("="*80)
        logger.error("Full traceback:")
        traceback.print_exc()
        logger.error("="*80 + "\n")
        sys.exit(1)
    
    finally:
        # Ensure lock is released
        lock.release()


if __name__ == "__main__":
    main()
