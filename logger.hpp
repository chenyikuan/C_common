#pragma once

#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#    include <direct.h>
#    include <io.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

/* must define before spdlog.h */
#ifndef SPDLOG_TRACE_ON
#    define SPDLOG_TRACE_ON
#endif

#ifndef SPDLOG_DEBUG_ON
#    define SPDLOG_DEBUG_ON
#endif

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "CSingleton.h"

#ifdef _WIN32
#    define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1) : __FILE__)
#else
#    define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)
#endif

#ifdef WIN32
#    define errcode WSAGetLastError()
#endif

#define criticalif(b, ...)                                            \
    do                                                                \
    {                                                                 \
        if ((b))                                                      \
        {                                                             \
            Logger::GetInstance().GetLogger()->critical(__VA_ARGS__); \
        }                                                             \
    } while (0)

/* define log suffix: [filename] [function:line] macro */
#define SUFFIX(msg)                       \
    std::string("[")                      \
        .append(__FILENAME__)             \
        .append("] [")                    \
        .append(__func__)                 \
        .append(":")                      \
        .append(std::to_string(__LINE__)) \
        .append("] ")                     \
        .append(msg)                      \
        .c_str()

template <typename T>
static std::string to_string(const T &n)
{
    std::ostringstream stm;
    stm << n;
    return stm.str();
}

static std::string GetHomeDirectory()
{
#ifdef _WIN32
    char *homeEnv = getenv("HOME");
    std::string homeDir;
    if (homeEnv == nullptr)
    {
        /// todo
        // homeDir.append(getpwuid(getuid())->pw_dir);
    }
    else
    {
        homeDir.append(homeEnv);
    }
#else
    std::string homeDir(getenv("HOME"));
#endif

    return homeDir;
}

static bool CreateDirectory(const char *dir)
{
    assert(dir != NULL);
    if (access(dir, 0) == -1)
    {
#ifdef _WIN32
        int flag = mkdir(dir);
#else
        int flag = mkdir(dir, 0777);
#endif
        if (flag == 0)
        {
            return true;
        }
        else
        {
            std::cout << "create directory:" << dir << " failed." << std::endl;
            return false;
        }
    }

    return true;
}

class MyLogger: public CSingleton<MyLogger>
{
  public:
    // static MyLogger &GetInstance()
    // {
    //     static MyLogger m_instance;
    //     return m_instance;
    // }

    auto GetLogger()
    {
        return m_logger;
    }

    void SetTraceLevel()
    {
        m_logger->set_level(spdlog::level::trace);
    }

    void SetDebugLevel()
    {
        m_logger->set_level(spdlog::level::debug);
    }

    void SetInfoLevel()
    {
        m_logger->set_level(spdlog::level::info);
    }

    void SetWarnLevel()
    {
        m_logger->set_level(spdlog::level::warn);
    }

    void SetErrorLevel()
    {
        m_logger->set_level(spdlog::level::err);
    }

    void SetOffLevel()
    {
        m_logger->set_level(spdlog::level::off);
    }

  private:
    friend class CSingleton<MyLogger>;
    MyLogger()
    {
        CreateDirectory("logs");
        // set async log
        // spdlog::set_async_mode(32768);  // power of 2
        std::vector<spdlog::sink_ptr> sinkList;

        // // Create a file rotating logger with 40Mb size max and 5 rotated files
        // auto rotating = std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/logs.log", 1024 * 1024 * 40, 5, false);
        // rotating->set_level(spdlog::level::debug);
        // rotating->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%8l%$][thread:%t] %v");
        // sinkList.push_back(rotating);

        // Create a daily logger - a new file is created every day on 2:30am
        auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/logs.log", 2, 30);
        // daily_sink->set_level(spdlog::level::debug);
        daily_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%^%l%$] %v");
        sinkList.push_back(daily_sink);
        auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt >();
        stdout_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%^%l%$] %v");
        sinkList.push_back(stdout_sink);

        // spdlog::init_thread_pool(8192, 1);
        // m_logger = std::make_shared<spdlog::async_logger>("both", begin(sinkList), end(sinkList), spdlog::thread_pool());
        m_logger = std::make_shared<spdlog::logger>("cyk", begin(sinkList), end(sinkList));
        // register it if you need to access it globally
        spdlog::register_logger(m_logger);

        // set log level
        m_logger->set_level(spdlog::level::trace);

        // when an error occurred, flush disk immediately
        m_logger->flush_on(spdlog::level::trace);
        spdlog::flush_every(std::chrono::seconds(2));
    }

    ~MyLogger()
    {
        spdlog::drop_all();
    }

    MyLogger(const MyLogger &) = delete;
    MyLogger &operator=(const MyLogger &) = delete;

  private:
    // std::shared_ptr<spdlog::async_logger> m_logger;
    std::shared_ptr<spdlog::logger> m_logger;
};

// #define ALOG_LOGGER MyLogger::GetInstance()
// #define MYLOG_TRACE(msg, ...) MyLogger::GetInstance().GetLogger()->trace(SUFFIX(msg), ##__VA_ARGS__)
// #define MYLOG_DEBUG(msg, ...) MyLogger::GetInstance().GetLogger()->debug(SUFFIX(msg), ##__VA_ARGS__)
// #define MYLOG_INFO(msg, ...) MyLogger::GetInstance().GetLogger()->info(SUFFIX(msg), ##__VA_ARGS__)
// #define MYLOG_WARN(msg, ...) MyLogger::GetInstance().GetLogger()->warn(SUFFIX(msg), ##__VA_ARGS__)
// #define MYLOG_ERROR(msg, ...) MyLogger::GetInstance().GetLogger()->error(SUFFIX(msg), ##__VA_ARGS__)
// #define MYLOG_CRITICAL(msg, ...) MyLogger::GetInstance().GetLogger()->critical(SUFFIX(msg), ##__VA_ARGS__)

#define MYLOG_TRACE(msg, ...) MyLogger::get_instance()->GetLogger()->trace(SUFFIX(msg), ##__VA_ARGS__)
#define MYLOG_DEBUG(msg, ...) MyLogger::get_instance()->GetLogger()->debug(SUFFIX(msg), ##__VA_ARGS__)
#define MYLOG_INFO(msg, ...) MyLogger::get_instance()->GetLogger()->info(SUFFIX(msg), ##__VA_ARGS__)
#define MYLOG_WARN(msg, ...) MyLogger::get_instance()->GetLogger()->warn(SUFFIX(msg), ##__VA_ARGS__)
#define MYLOG_ERROR(msg, ...) MyLogger::get_instance()->GetLogger()->error(SUFFIX(msg), ##__VA_ARGS__)
#define MYLOG_CRITICAL(msg, ...) MyLogger::get_instance()->GetLogger()->critical(SUFFIX(msg), ##__VA_ARGS__)
