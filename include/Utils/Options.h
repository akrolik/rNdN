#pragma once

#include "Libraries/cxxopts.hpp"

namespace Utils {

class Options
{
public:
	static constexpr char const *Opt_Help = "help";
	static constexpr char const *Opt_Optimize = "optimize";
	static constexpr char const *Opt_Print_hir = "print-hir";
	static constexpr char const *Opt_Print_symbol = "print-symbol";
	static constexpr char const *Opt_Print_analysis = "print-analysis";
	static constexpr char const *Opt_Print_outline = "print-hir-outline";
	static constexpr char const *Opt_Print_ptx = "print-ptx";
	static constexpr char const *Opt_Print_json = "print-json";

	Options(Options const&) = delete;
	void operator=(Options const&) = delete;

	static void Initialize(int argc, const char *argv[])
	{
		auto& instance = GetInstance();
		auto results = instance.m_options.parse(argc, argv);
		if (results.count(Opt_Help) > 0)
		{
			std::cout << instance.m_options.help({"", "Debug"}) << std::endl;
			std::exit(EXIT_SUCCESS);
		}
		instance.m_results = results;
	}

	static bool Present(const std::string& name)
	{
		return GetInstance().m_results.count(name) > 0;
	}

	template<typename T = bool>
	static const T& Get(const std::string& name)
	{
		return GetInstance().m_results[name].as<T>();
	}

private:
	Options() : m_options("r3d3", "Optimizing JIT compiler for HorseIR targetting PTX")
	{
		m_options.add_options()
			("h,help", "Display this help menu")
			("O,optimize", "Enable PTX optimizer")
		;
		m_options.add_options("Debug Printing (stdout)")
			("print-hir", "Pretty print input HorseIR program")
			("print-symbol", "Print symbol table")
			("print-analysis", "Print analyses")
			("print-hir-outline", "Pretty print outlined HorseIR program")
			("print-ptx", "Print generated PTX code")
			("print-json", "Print generated PTX JSON")
		;
	}

	static Options& GetInstance()
	{
		static Options instance;
		return instance;
	}

	cxxopts::Options m_options;
	cxxopts::ParseResult m_results;
};

}
