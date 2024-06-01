#pragma once

#include <iostream>
#include <string>
#include <chrono>

class Timer
{
public:
    Timer()
    {
        Reset();
    }

    void Reset()
    {
        m_startTimepoint = std::chrono::high_resolution_clock::now();
    }

    float Elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - m_startTimepoint).count() * 1e-9f;
    }

    float ElapsedMS() const
    {
        return Elapsed() * 1000.0f;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimepoint;
};

class TimerScope
{
public:
    TimerScope(const std::string& name)
            : m_name(name) {}

    ~TimerScope()
    {
        const float time = m_timer.ElapsedMS();
        std::cout << "[TIMER] " << m_name << ": " << time << "ms" << std::endl;

    }

private:
    std::string m_name;
    Timer m_timer;
};
