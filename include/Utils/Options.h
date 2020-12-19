#pragma once

#include "Libraries/cxxopts.hpp"

#include "Utils/Logger.h"

namespace Utils {

class Options
{
public:
	static constexpr char const *Opt_Help = "help";

	static constexpr char const *Opt_Optimize_outline = "optimize-outline";
	static constexpr char const *Opt_Optimize_hir = "optimize-hir";
	static constexpr char const *Opt_Optimize_ptx = "optimize-ptx";
	static constexpr char const *Opt_Optimize_sass = "optimize-sass";

	static constexpr char const *Opt_Debug_load = "debug-load";
	static constexpr char const *Opt_Debug_print = "debug-print";
	static constexpr char const *Opt_Debug_time = "debug-time";

	static constexpr char const *Opt_Frontend_print_hir = "frontend-print-hir";
	static constexpr char const *Opt_Frontend_print_hir_typed = "frontend-print-hir-typed";
	static constexpr char const *Opt_Frontend_print_symbols = "frontend-print-symbols";
	static constexpr char const *Opt_Frontend_print_analysis = "frontend-print-analysis";
	static constexpr char const *Opt_Frontend_print_outline = "frontend-print-outline";
	static constexpr char const *Opt_Frontend_print_outline_graph = "frontend-print-outline-graph";
	static constexpr char const *Opt_Frontend_print_ptx = "frontend-print-ptx";
	static constexpr char const *Opt_Frontend_print_json = "frontend-print-json";

	static constexpr char const *Opt_Backend = "backend";
	static constexpr char const *Opt_Backend_dump_elf = "backend-dump-elf";
	static constexpr char const *Opt_Backend_print_analysis = "backend-print-analysis";
	static constexpr char const *Opt_Backend_print_analysis_block = "backend-print-analysis-block";
	static constexpr char const *Opt_Backend_print_cfg = "backend-print-cfg";
	static constexpr char const *Opt_Backend_print_sass = "backend-print-sass";
	static constexpr char const *Opt_Backend_print_assembled = "backend-print-assembled";
	static constexpr char const *Opt_Backend_print_elf = "backend-print-elf";

	static constexpr char const *Opt_Algo_reduction = "algo-reduction";
	static constexpr char const *Opt_Algo_sort = "algo-sort";
	static constexpr char const *Opt_Algo_group = "algo-group";
	static constexpr char const *Opt_Algo_join = "algo-join";
	static constexpr char const *Opt_Algo_hash_size = "algo-hash-size";
	static constexpr char const *Opt_Algo_like = "algo-like";
	static constexpr char const *Opt_Algo_unique = "algo-unique";

	static constexpr char const *Opt_Data_load_tpch = "data-load-tpch";
	static constexpr char const *Opt_Data_path_tpch = "data-path-tpch";
	static constexpr char const *Opt_Data_resize = "data-resize";
	static constexpr char const *Opt_Data_allocator = "data-allocator";
	static constexpr char const *Opt_Data_page_size = "data-page-size";
	static constexpr char const *Opt_Data_page_count = "data-page-count";

	static constexpr char const *Opt_File = "file";

	Options(Options const&) = delete;
	void operator=(Options const&) = delete;

	static void Initialize(int argc, const char *argv[])
	{
		auto& instance = GetInstance();
		auto results = instance.m_options.parse(argc, argv);
		if (results.count(Opt_Help) > 0)
		{
			Utils::Logger::LogInfo(instance.m_options.help({"", "Optimization", "Debug", "Frontend", "Backend", "Algorithm", "Data"}), 0, true, Utils::Logger::NoPrefix);
			std::exit(EXIT_SUCCESS);
		}
		instance.m_results = results;
	}

	// Optimization

	enum class OutlineOptimization {
		None,
		Flow,
		Full
	};

	static OutlineOptimization GetOptimize_Outline()
	{
		auto optMode = Get<std::string>(Opt_Optimize_outline);
		if (optMode == "none")
		{
			return OutlineOptimization::None;
		}
		else if (optMode == "flow")
		{
			return OutlineOptimization::Flow;
		}
		else if (optMode == "full")
		{
			return OutlineOptimization::Full;
		}
		Utils::Logger::LogError("Unknown outline optimization '" + optMode + "'");
	}

	static bool IsOptimize_HorseIR() { return Present(Opt_Optimize_hir); }
	static bool IsOptimize_PTX() { return Present(Opt_Optimize_ptx); }
	static bool IsOptimize_SASS() { return Present(Opt_Optimize_sass); }

	// Debug

	static bool IsDebug_Load() { return Present(Opt_Debug_load); }
	static bool IsDebug_Print() { return Present(Opt_Debug_print); }
	static bool IsDebug_Time() { return Present(Opt_Debug_time); }

	// Frontend

	static bool IsFrontend_PrintHorseIR() { return Present(Opt_Frontend_print_hir); }
	static bool IsFrontend_PrintHorseIRTyped() { return Present(Opt_Frontend_print_hir_typed); }
	static bool IsFrontend_PrintSymbols() { return Present(Opt_Frontend_print_symbols); }
	static bool IsFrontend_PrintAnalysis() { return Present(Opt_Frontend_print_analysis); }
	static bool IsFrontend_PrintOutline() { return Present(Opt_Frontend_print_outline); }
	static bool IsFrontend_PrintOutlineGraph() { return Present(Opt_Frontend_print_outline_graph); }
	static bool IsFrontend_PrintPTX() { return Present(Opt_Frontend_print_ptx); }
	static bool IsFrontend_PrintJSON() { return Present(Opt_Frontend_print_json); }

	// Backend

	enum class BackendKind {
		ptxas,
		r3d3
	};

	static BackendKind GetBackendKind()
	{
		auto backend = Get<std::string>(Opt_Backend);
		if (backend == "ptxas")
		{
			return BackendKind::ptxas;
		}
		else if (backend == "r3d3")
		{
			return BackendKind::r3d3;
		}
		Utils::Logger::LogError("Unknown backend '" + backend + "'");
	}

	static bool IsBackend_DumpELF() { return Present(Opt_Backend_dump_elf); }
	static bool IsBackend_PrintAnalysis() { return Present(Opt_Backend_print_analysis); }
	static bool IsBackend_PrintAnalysisBlock() { return Present(Opt_Backend_print_analysis_block); }
	static bool IsBackend_PrintCFG() { return Present(Opt_Backend_print_cfg); }
	static bool IsBackend_PrintSASS() { return Present(Opt_Backend_print_sass); }
	static bool IsBackend_PrintAssembled() { return Present(Opt_Backend_print_assembled); }
	static bool IsBackend_PrintELF() { return Present(Opt_Backend_print_elf); }

	// Algorithm

	enum class ReductionKind {
		ShuffleBlock,
		ShuffleWarp,
		Shared
	};

	static ReductionKind GetAlgorithm_ReductionKind()
	{
		auto reductionMode = Get<std::string>(Opt_Algo_reduction);
		if (reductionMode == "shflwarp")
		{
			return ReductionKind::ShuffleWarp;
		}
		else if (reductionMode  == "shflblock")
		{
			return ReductionKind::ShuffleBlock;
		}
		else if (reductionMode == "shared")
		{
			return ReductionKind::Shared;
		}
		Utils::Logger::LogError("Unknown reduction mode '" + reductionMode + "'");
	}

	enum class SortKind {
		GlobalSort,
		SharedSort
	};

	static SortKind GetAlgorithm_SortKind()
	{
		auto sortMode = Get<std::string>(Opt_Algo_sort);
		if (sortMode == "global")
		{
			return SortKind::GlobalSort;
		}
		else if (sortMode == "shared")
		{
			return SortKind::SharedSort;
		}
		Utils::Logger::LogError("Unknown sort mode '" + sortMode + "'");
	}

	enum class GroupKind {
		CellGroup,
		CompressedGroup
	};

	static GroupKind GetAlgorithm_GroupKind()
	{
		auto groupMode = Get<std::string>(Opt_Algo_group);
		if (groupMode == "cell")
		{
			return GroupKind::CellGroup;
		}
		else if (groupMode == "compressed")
		{
			return GroupKind::CompressedGroup;
		}
		Utils::Logger::LogError("Unknown group mode '" + groupMode + "'");
	}

	enum class JoinKind {
		LoopJoin,
		HashJoin
	};

	static JoinKind GetAlgorithm_JoinKind()
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

	static unsigned int GetAlgorithm_HashSize() { return Get<unsigned int>(Opt_Algo_hash_size); }

	enum class LikeKind {
		PCRELike,
		OptLike
	};

	static LikeKind GetAlgorithm_LikeKind()
	{
		auto likeMode = Get<std::string>(Opt_Algo_like);
		if (likeMode == "pcre")
		{
			return LikeKind::PCRELike;
		}
		else if (likeMode == "opt")
		{
			return LikeKind::OptLike;
		}
		Utils::Logger::LogError("Unknown like mode '" + likeMode + "'");
	}

	enum class UniqueKind {
		SortUnique,
		LoopUnique
	};

	static UniqueKind GetAlgorithm_UniqueKind()
	{
		auto uniqueKind = Get<std::string>(Opt_Algo_unique);
		if (uniqueKind == "sort")
		{
			return UniqueKind::SortUnique;
		}
		else if (uniqueKind == "loop")
		{
			return UniqueKind::LoopUnique;
		}
		Utils::Logger::LogError("Unknown unique mode '" + uniqueKind + "'");
	}

	// Data

	static bool IsData_LoadTPCH() { return Present(Opt_Data_load_tpch); }
	static std::string GetData_PathTPCH() { return Get<std::string>(Opt_Data_path_tpch); }

	static float GetData_Resize() { return Get<float>(Opt_Data_resize); }

	enum class AllocatorKind {
		CUDA,
		Linear
	};

	static AllocatorKind GetData_AllocatorKind()
	{
		auto allocator = Get<std::string>(Opt_Data_allocator);
		if (allocator == "cuda")
		{
			return AllocatorKind::CUDA;
		}
		else if (allocator == "linear")
		{
			return AllocatorKind::Linear;
		}
		Utils::Logger::LogError("Unknown allocator '" + allocator + "'");
	}

	static unsigned long long GetData_PageSize() { return Get<unsigned long long>(Opt_Data_page_size); }
	static unsigned int GetData_PageCount() { return Get<unsigned int>(Opt_Data_page_count); }

	// Input file

	static bool HasInputFile() { return Present(Opt_File); }
	static std::string GetInputFile() { return Get<std::string>(Opt_File); }

private:
	// Getter

	static bool Present(const std::string& name)
	{
		return GetInstance().m_results.count(name) > 0;
	}

	template<typename T = bool>
	static const T& Get(const std::string& name)
	{
		return GetInstance().m_results[name].as<T>();
	}

	// Initialization

	Options() : m_options("r3d3", "Optimizing JIT compiler for HorseIR targetting PTX")
	{
		m_options.add_options()
			("h,help", "Display this help menu")
		;
		m_options.add_options("Optimization")
			(Opt_Optimize_outline, "Outline graph optimization [none|flow|full]", cxxopts::value<std::string>()->default_value("full"))
			(Opt_Optimize_ptx, "PTX optimizer")
			(Opt_Optimize_sass, "SASS optimizer")
		;
		m_options.add_options("Debug")
			(Opt_Debug_load, "Debug data loading")
			(Opt_Debug_print, "Print debug logs")
			(Opt_Debug_time, "Print executing timings")
		;
		m_options.add_options("Frontend")
			(Opt_Frontend_print_hir, "Pretty print input HorseIR program")
			(Opt_Frontend_print_hir_typed, "Pretty print typed HorseIR program")
			(Opt_Frontend_print_symbols, "Print symbol table")
			(Opt_Frontend_print_analysis, "Print frontend analyses")
			(Opt_Frontend_print_outline, "Pretty print outlined HorseIR program")
			(Opt_Frontend_print_outline_graph, "Pretty print outliner graph")
			(Opt_Frontend_print_ptx, "Print generated PTX code")
			(Opt_Frontend_print_json, "Print generated PTX JSON")
		;
		m_options.add_options("Backend")
			(Opt_Backend, "Backend assembler [ptxas|r3d3]", cxxopts::value<std::string>()->default_value("ptxas"))
			(Opt_Backend_dump_elf, "Dump assembled .cubin ELF file")
			(Opt_Backend_print_analysis, "Print backend analyses")
			(Opt_Backend_print_analysis_block, "Print backend analyses in basic blocks mode")
			(Opt_Backend_print_cfg, "Print control-flow graph")
			(Opt_Backend_print_sass, "Print generated SASS code")
			(Opt_Backend_print_assembled, "Print assembled SASS code")
			(Opt_Backend_print_elf, "Print generated ELF file")
		;
		m_options.add_options("Algorithm")
			(Opt_Algo_reduction, "Reduction [sfhlwarp|shflblock|shared]", cxxopts::value<std::string>()->default_value("shflwarp"))
			(Opt_Algo_sort, "Sort mode [global|shared]", cxxopts::value<std::string>()->default_value("shared"))
			(Opt_Algo_group, "Group mode [cell|compressed]", cxxopts::value<std::string>()->default_value("compressed"))
			(Opt_Algo_join, "Join mode [loop|hash]", cxxopts::value<std::string>()->default_value("hash"))
			(Opt_Algo_hash_size, "Hash table size [data * 2^n]", cxxopts::value<unsigned int>()->default_value("1"))
			(Opt_Algo_like, "Like mode [pcre|opt]", cxxopts::value<std::string>()->default_value("opt"))
			(Opt_Algo_unique, "Unique mode [sort|loop]", cxxopts::value<std::string>()->default_value("loop"))
		;
		m_options.add_options("Backend")
		;
		m_options.add_options("Data")
			(Opt_Data_load_tpch, "Load TPC-H data")
			(Opt_Data_path_tpch, "TPC-H data path", cxxopts::value<std::string>())
			(Opt_Data_allocator, "GPU allocator algorithm [cuda|linear]", cxxopts::value<std::string>()->default_value("linear"))
			(Opt_Data_page_size, "GPU page size", cxxopts::value<unsigned long long>()->default_value("2147483648"))
			(Opt_Data_page_count, "GPU page count", cxxopts::value<unsigned int>()->default_value("2"))
			(Opt_Data_resize, "Resize buffer factor (only used with cuda allocator)", cxxopts::value<float>()->default_value("0.9"))
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
