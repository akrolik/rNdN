#pragma once

#include "Libraries/cxxopts.hpp"

namespace Utils {

class Options
{
public:
	static constexpr char const *Opt_Help = "help";
	static constexpr char const *Opt_Optimize = "optimize";
	static constexpr char const *Opt_Print_debug = "print-debug";
	static constexpr char const *Opt_Print_hir = "print-hir";
	static constexpr char const *Opt_Print_hir_typed = "print-hir-typed";
	static constexpr char const *Opt_Print_symbol = "print-symbol";
	static constexpr char const *Opt_Print_analysis = "print-analysis";
	static constexpr char const *Opt_Print_outline = "print-outline";
	static constexpr char const *Opt_Print_outline_graph = "print-outline-graph";
	static constexpr char const *Opt_Print_ptx = "print-ptx";
	static constexpr char const *Opt_Print_json = "print-json";
	static constexpr char const *Opt_Load_tpch = "load-tpch";

	static constexpr char const *Opt_File = "file";

	Options(Options const&) = delete;
	void operator=(Options const&) = delete;

	static void Initialize(int argc, const char *argv[])
	{
		auto& instance = GetInstance();
		auto results = instance.m_options.parse(argc, argv);
		if (results.count(Opt_Help) > 0)
		{
			std::cout << instance.m_options.help({"", "Debug", "Data"}) << std::endl;
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
			(Opt_Print_debug, "Print debug logs (excludes below)")
			(Opt_Print_hir, "Pretty print input HorseIR program")
			(Opt_Print_hir_typed, "Pretty print typed HorseIR program")
			(Opt_Print_symbol, "Print symbol table")
			(Opt_Print_analysis, "Print analyses")
			(Opt_Print_outline, "Pretty print outlined HorseIR program")
			(Opt_Print_outline_graph, "Pretty print outliner graphs")
			(Opt_Print_ptx, "Print generated PTX code")
			(Opt_Print_json, "Print generated PTX JSON")
		;
		m_options.add_options("Data")
			(Opt_Load_tpch, "Load TPC-H data")
		;
		m_options.add_options("Query")
			(Opt_File, "Query HorseIR file", cxxopts::value<std::string>())
		;
		m_options.parse_positional(Opt_File);
		m_options.positional_help("filename");
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
