#pragma once

#include "Libraries/cxxopts.hpp"

namespace Utils {

class Options
{
public:
	static constexpr char const *Opt_Help = "help";
	static constexpr char const *Opt_Optimize = "optimize";
	static constexpr char const *Opt_Dump_hir = "dump-hir";
	static constexpr char const *Opt_Dump_symbol = "dump-symbol";
	static constexpr char const *Opt_Dump_type = "dump-type";
	static constexpr char const *Opt_Dump_shape = "dump-shape";
	static constexpr char const *Opt_Dump_ptx = "dump-ptx";
	static constexpr char const *Opt_Dump_json = "dump-json";

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
		m_options.add_options("Debug")
			("i,dump-hir", "Dump pretty printed input HorseIR to stdout")
			("s,dump-symbol", "Dump symbol table to stdout")
			("t,dump-type", "Dump type analysis result to stdout")
			("a,dump-shape", "Dump shape analysis result to stdout")
			("p,dump-ptx", "Dump generated PTX code to stdout")
			("j,dump-json", "Dump generated PTX JSON to stdout")
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
