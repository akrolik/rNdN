#include <iostream>

namespace Utils {

class Logger
{
public:
	Logger(Logger const&) = delete;
	void operator=(Logger const&) = delete;

	static void LogInfo(const std::string& info, const std::string& prefix = "[INFO] ")
	{
		std::cout << prefix << info << std::endl;
	}

	static void LogError(const std::string& error, const std::string& prefix = "[ERROR] ", bool exit = true)
	{
		std::cerr << prefix << error << std::endl;
		if (exit)
		{
			std::exit(EXIT_FAILURE);
		}
	}

private:
	Logger() {}

	static Logger& GetInstance()
	{
		static Logger instance;
		return instance;
	}

};

}
