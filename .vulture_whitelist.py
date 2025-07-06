# Vulture whitelist for legitimate unused code
# This file whitelists intentionally unused code to avoid false positives

# Models and classes that are used via reflection or external APIs
_.AskHandler
_.DeepAskHandler
_.TokenValidator
_.RequestValidator
_.MergeConflict
_.FileNode
_.CodebaseSizeInfo

# Methods that are part of middleware/API interfaces
_.dispatch
_.format_response
_.format_error_response
_.send_event
_.validate_bearer_token
_.generate_token
_.mask_token

# Handler methods used by CLI/API
_.ask_question
_.deepask_question
_.handle_mcp_request
_.server_sent_events

# Database methods used by higher-level abstractions
_.get_cached_token_count
_.cache_token_count
_.cleanup_expired_cache
_.get_connection_with_retry

# Cleanup and maintenance methods
_.mark_files_for_cleanup
_.restore_file_from_cleanup
_.get_files_to_be_cleaned
_.get_cleanup_stats
_.cleanup_old_entries

# Statistics and monitoring methods
_.get_rate_limit_stats
_.record_request
_.reset_metrics
_.list_active_databases

# Error handling functions
_.handle_database_errors
_.handle_file_errors
_.validate_arguments
_.require_fields
_.handle_file_operations
_.handle_database_operations

# Test utilities and fixtures
_.event_loop
_.sample_file_descriptions
_.setup_test_logging
_.mock_git_repo
_.large_file_descriptions
_.performance_markers
_.assert_file_description_equal
_.create_test_file_description
_.async_test_context
_.sample_file_description

# Logging and validation methods
_.log_database_metrics
_.validate_json_size
_.validate_mcp_request
_.sanitize_user_input

# Constants and variables that may be used by external code
_.NETWORK
_.AUTHENTICATION
_.PERMISSION
_.authenticated
_.jsonrpc
_.exc_type
_.exc_val
_.exc_tb
_.operation_count
_.connection_pool_size
_.base_path
_.all_files_count
_.cleaned_up_files
_.source_branch
_.target_branch
_.source_description
_.target_description
_.resolution
_.row_factory

# Private methods that are part of internal APIs
_._escape_quotes_in_term
_._parse_json_robust

# Resource management methods
_.add_resource
_.get_remote_url
_.get_current_branch
_.list_branches
