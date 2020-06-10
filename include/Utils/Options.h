#pragma once

#include "Libraries/cxxopts.hpp"

namespace Utils {

class Options
{
public:
	static constexpr char const *Opt_Help = "help";
	static constexpr char const *Opt_Optimize = "optimize";
	static constexpr char const *Opt_Optimize_outline = "optimize-outline";

	static constexpr char const *Opt_Print_debug = "print-debug";
	static constexpr char const *Opt_Print_hir = "print-hir";
	static constexpr char const *Opt_Print_hir_typed = "print-hir-typed";
	static constexpr char const *Opt_Print_symbol = "print-symbol";
	static constexpr char const *Opt_Print_analysis = "print-analysis";
	static constexpr char const *Opt_Print_outline = "print-outline";
	static constexpr char const *Opt_Print_outline_graph = "print-outline-graph";
	static constexpr char const *Opt_Print_ptx = "print-ptx";
	static constexpr char const *Opt_Print_json = "print-json";
	static constexpr char const *Opt_Print_time = "print-time";

	static constexpr char const *Opt_Algo_reduction = "algo-reduction";
	static constexpr char const *Opt_Algo_smem_sort = "algo-smem-sort";
	static constexpr char const *Opt_Algo_group_compressed = "algo-group-compressed";
	static constexpr char const *Opt_Algo_join = "algo-join";
	static constexpr char const *Opt_Algo_hash_size = "algo-hash-size";

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
			std::cout << instance.m_options.help({"", "Debug", "Optimization", "Algorithm", "Data"}) << std::endl;
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

	enum class JoinKind {
		LoopJoin,
		HashJoin
	};

	static JoinKind GetJoinKind()
	{
		auto joinMode = Get<std::string>(Opt_Algo_join);
		if (joinMode == "loop")
		{
			return JoinKind::LoopJoin;
		}
		else if (joinMode == "hash")
		{
			return JoinKind::HashJoin;
		}
		Utils::Logger::LogError("Unknown join mode '" + joinMode + "'");
	}

private:
	Options() : m_options("r3d3", "Optimizing JIT compiler for HorseIR targetting PTX")
	{
		m_options.add_options()
			("h,help", "Display this help menu")
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
			(Opt_Print_time, "Print timings")
		;
		m_options.add_options("Optimization")
			("O,optimize", "Unimplemented")
			(Opt_Optimize_outline, "Enable outliner graph optimization", cxxopts::value<bool>()->default_value("true"))
		;
		m_options.add_options("Algorithm")
			(Opt_Algo_reduction, "Reduction [sfhlwarp|shflblock|shared]", cxxopts::value<std::string>()->default_value("shflwarp"))
			(Opt_Algo_smem_sort, "Shared memory sort", cxxopts::value<bool>()->default_value("true"))
			(Opt_Algo_group_compressed, "Group compressed list", cxxopts::value<bool>()->default_value("true"))
			(Opt_Algo_join, "Join mode [loop|hash]", cxxopts::value<std::string>()->default_value("loop"))
			(Opt_Algo_hash_size, "Hash table size [data * 2^n]", cxxopts::value<unsigned int>()->default_value("1"))
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
