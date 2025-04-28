import logging
import sys
from pathlib import Path
import datetime
import json
import os


class SimulationLogFormatter(logging.Formatter):
    """Custom formatter for simulation logs with color coding for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41;37m',  # White on red background
        'RESET': '\033[0m',  # Reset to default
    }

    def format(self, record):
        """Format log record with appropriate colors."""
        log_message = super().format(record)

        # Add colors in terminal environments, skip for file logging
        if sys.stdout.isatty():  # Check if output is terminal
            level_name = record.levelname
            if level_name in self.COLORS:
                return f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"

        return log_message


def setup_logging(log_file=None, level=logging.INFO, console_level=None):
    """
    Configure logging for the simulation.

    Parameters:
    -----------
    log_file : str or Path, optional
        Path to the log file
    level : int, optional
        Logging level for file handler
    console_level : int, optional
        Logging level for console handler, defaults to level if None

    Returns:
    --------
    logging.Logger
        Configured root logger
    """
    if console_level is None:
        console_level = level

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs at root level

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = SimulationLogFormatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Log system info at startup
    root_logger.info("=" * 80)
    root_logger.info(f"Logging initialized at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    root_logger.info(f"Log file: {log_file if log_file else 'Console only'}")

    # Log Python version and key packages
    root_logger.debug(f"Python version: {sys.version}")
    try:
        import openmc
        root_logger.debug(f"OpenMC version: {openmc.__version__}")
    except (ImportError, AttributeError):
        root_logger.debug("OpenMC version: Unknown")

    root_logger.debug(f"Process ID: {os.getpid()}")
    root_logger.info("=" * 80)

    return root_logger


def log_configuration(config, logger=None):
    """
    Log configuration parameters.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger to use, defaults to root logger
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("Configuration summary:")
    try:
        # Log key configuration parameters
        if 'geometry' in config:
            geom = config['geometry']
            logger.info(f"  Wall thickness: {geom.get('wall_thickness')} ft")
            logger.info(f"  Channel diameter: {geom.get('channel_diameter')} cm")
            logger.info(f"  Source distance: {geom.get('source_distance')} ft")

        if 'source' in config:
            src = config['source']
            logger.info(f"  Source energy range: {src.get('energy_range')} MeV")

        if 'simulation' in config:
            sim = config['simulation']
            logger.info(f"  Particles: {sim.get('particles')}")
            logger.info(f"  Batches: {sim.get('batches')}")

        # Log full configuration in debug level
        logger.debug("Full configuration:")
        logger.debug(json.dumps(config, indent=2))

    except Exception as e:
        logger.warning(f"Error logging configuration: {str(e)}")


class SimulationProgressLogger:
    """Helper class to log simulation progress."""

    def __init__(self, total_batches, log_frequency=5, logger=None):
        """
        Initialize progress logger.

        Parameters:
        -----------
        total_batches : int
            Total number of batches to simulate
        log_frequency : int, optional
            How often to log progress (in percentage or batches)
        logger : logging.Logger, optional
            Logger to use, defaults to root logger
        """
        self.total_batches = total_batches
        self.log_frequency = log_frequency
        self.logger = logger or logging.getLogger()
        self.start_time = datetime.datetime.now()
        self.last_log_time = self.start_time
        self.last_log_batch = 0

    def update(self, current_batch):
        """
        Update progress and log if needed.

        Parameters:
        -----------
        current_batch : int
            Current batch number
        """
        # Calculate progress
        progress = (current_batch / self.total_batches) * 100

        # Determine if we should log based on frequency
        log_threshold = False

        # Log based on percentage increments
        if isinstance(self.log_frequency, int):
            current_percent = int(progress)
            last_percent = int((self.last_log_batch / self.total_batches) * 100)
            log_threshold = (current_percent // self.log_frequency) > (last_percent // self.log_frequency)

        # Always log the first and last batch
        if current_batch == 1 or current_batch == self.total_batches:
            log_threshold = True

        if log_threshold:
            now = datetime.datetime.now()
            elapsed = (now - self.start_time).total_seconds()

            # Calculate ETA
            if current_batch > 1:
                eta_seconds = (elapsed / current_batch) * (self.total_batches - current_batch)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "calculating..."

                self.logger.info(
                    f"Progress: {progress:.1f}% (Batch {current_batch}/{self.total_batches}) - Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} - ETA: {eta_str}")

                # Update tracking variables
                self.last_log_time = now
                self.last_log_batch = current_batch

            def finalize(self):
                """Log final statistics when simulation is complete."""
                end_time = datetime.datetime.now()
                total_time = (end_time - self.start_time).total_seconds()

                self.logger.info("=" * 60)
                self.logger.info(f"Simulation completed")
                self.logger.info(f"Total time: {str(datetime.timedelta(seconds=int(total_time)))}")
                self.logger.info(f"Average time per batch: {total_time / self.total_batches:.2f} seconds")
                self.logger.info("=" * 60)

        def capture_warnings():
            """Capture Python warnings and redirect them to the logging system."""
            import warnings

            # Create a capture class for handling warnings
            class WarningToLogger:
                def __init__(self, logger):
                    self.logger = logger

                def __call__(self, message, category, filename, lineno, file=None, line=None):
                    self.logger.warning(f"{category.__name__}: {message} (in {filename}, line {lineno})")

            # Create dedicated warnings logger
            warnings_logger = logging.getLogger("warnings")

            # Set warnings to be redirected to logging
            warnings.showwarning = WarningToLogger(warnings_logger)

        def log_exception(exc_info=None, logger=None):
            """
            Log an exception with full traceback.

            Parameters:
            -----------
            exc_info : tuple, optional
                Exception info as returned by sys.exc_info()
            logger : logging.Logger, optional
                Logger to use, defaults to root logger
            """
            import traceback

            if logger is None:
                logger = logging.getLogger()

            if exc_info is None:
                exc_info = sys.exc_info()

            if exc_info[0] is not None:  # If there's an actual exception
                logger.error(f"Exception occurred: {exc_info[1]}")
                tb_lines = traceback.format_exception(*exc_info)
                logger.error("".join(tb_lines))

