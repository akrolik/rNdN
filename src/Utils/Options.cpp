#include "Utils/Options.h"

#include "Utils/Logger.h"

namespace Utils {

void Options::Initialize(int argc, const char *argv[])
{
	auto& instance = GetInstance();
	auto results = instance.m_options.parse(argc, argv);
	if (results.count(Opt_Help) > 0)
	{
		Utils::Logger::LogInfo(instance.m_options.help({"", "Optimization", "Debug", "Frontend", "Backend", "Backend Scheduler", "Link", "Algorithm", "Data"}), 0, true, Utils::Logger::NoPrefix);
		std::exit(EXIT_SUCCESS);
	}
	instance.m_results = results;
}

// Optimization

Options::OutlineOptimization Options::GetOptimize_Outline()
{
	auto& optMode = Get<std::string>(Opt_Optimize_outline);
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

bool Options::IsOptimize_HorseIR()
{
	return Get(Opt_Optimize_hir);
}

bool Options::IsOptimize_PTX()
{
	return Get(Opt_Optimize_ptx);
}
bool Options::IsOptimize_SASS()
{
	return Get(Opt_Optimize_sass);
}

// Debug

bool Options::IsDebug_Load()
{
	return Get(Opt_Debug_load);
}

bool Options::IsDebug_Print()
{
	return Get(Opt_Debug_print);
}

bool Options::IsDebug_Time()
{
	return Get(Opt_Debug_time);
}

// Frontend

bool Options::IsFrontend_PrintHorseIR()
{
	return Get(Opt_Frontend_print_hir);
}

bool Options::IsFrontend_PrintHorseIRTyped()
{
	return Get(Opt_Frontend_print_hir_typed); }

bool Options::IsFrontend_PrintSymbols()
{
	return Get(Opt_Frontend_print_symbols);
}

bool Options::IsFrontend_PrintAnalysis(const std::string& analysis, const std::string& function)
{
	if (Present(Opt_Frontend_print_analysis))
	{
		if (Present(Opt_Frontend_print_analysis_func))
		{
			auto found = false;
			for (auto& opt : Get<std::vector<std::string>>(Opt_Frontend_print_analysis_func))
			{
				if (opt == function || opt == "all")
				{
					found = true;
				}
			}

			if (!found)
			{
				return false;
			}
		}

		for (auto& opt : Get<std::vector<std::string>>(Opt_Frontend_print_analysis))
		{
			if (opt == analysis || opt == "all")
			{
				return true;
			}
		}
	}
	return false;
}

bool Options::IsFrontend_PrintOutline()
{
	return Get(Opt_Frontend_print_outline);
}

bool Options::IsFrontend_PrintPTX()
{
	return Get(Opt_Frontend_print_ptx);
}

bool Options::IsFrontend_PrintJSON()
{
	return Get(Opt_Frontend_print_json);
}

// Backend

Options::BackendKind Options::GetBackend_Kind()
{
	auto& backend = Get<std::string>(Opt_Backend);
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

Options::BackendRegisterAllocator Options::GetBackend_RegisterAllocator()
{
	auto& allocator = Get<std::string>(Opt_Backend_reg_alloc);
	if (allocator == "virtual")
	{
		return BackendRegisterAllocator::Virtual;
	}
	else if (allocator == "linear")
	{
		return BackendRegisterAllocator::LinearScan;
	}
	Utils::Logger::LogError("Unknown register allocator '" + allocator + "'");
}

Options::BackendScheduler Options::GetBackend_Scheduler()
{
	auto& scheduler = Get<std::string>(Opt_Backend_scheduler);
	if (scheduler == "linear")
	{
		return BackendScheduler::Linear;
	}
	else if (scheduler == "list")
	{
		return BackendScheduler::List;
	}
	Utils::Logger::LogError("Unknown scheduler '" + scheduler + "'");
}

bool Options::IsBackend_InlineBranch()
{
	return Get(Opt_Backend_inline_branch);
}

unsigned int Options::GetBackend_InlineBranchThreshold()
{
	return Get<unsigned int>(Opt_Backend_inline_branch_threshold);
}

bool Options::IsBackend_DumpELF()
{
	return Get(Opt_Backend_dump_elf);
}

bool Options::IsBackend_PrintAnalysis(const std::string& analysis, const std::string& function)
{
	if (Present(Opt_Backend_print_analysis))
	{
		if (Present(Opt_Backend_print_analysis_func))
		{
			auto found = false;
			for (auto& opt : Get<std::vector<std::string>>(Opt_Backend_print_analysis_func))
			{
				if (opt == function || opt == "all")
				{
					found = true;
				}
			}

			if (!found)
			{
				return false;
			}
		}

		for (auto& opt : Get<std::vector<std::string>>(Opt_Backend_print_analysis))
		{
			if (opt == analysis || opt == "all")
			{
				return true;
			}
		}
	}
	return false;
}

bool Options::IsBackend_PrintAnalysisBlock()
{
	return Get(Opt_Backend_print_analysis_block);
}

bool Options::IsBackend_PrintCFG()
{
	return Get(Opt_Backend_print_cfg);
}

bool Options::IsBackend_PrintStructured()
{
	return Get(Opt_Backend_print_structured);
}

bool Options::IsBackend_PrintScheduled()
{
	return Get(Opt_Backend_print_scheduled);
}

bool Options::IsBackend_PrintSASS()
{
	return Get(Opt_Backend_print_sass);
}

bool Options::IsBackend_PrintAssembled()
{
	return Get(Opt_Backend_print_assembled);
}

bool Options::IsBackend_PrintELF()
{
	return Get(Opt_Backend_print_elf);
}

// Backend scheduler

bool Options::IsBackendSchedule_Dual()
{
	return Get(Opt_Backend_scheduler_dual);
}

bool Options::IsBackendSchedule_Reuse()
{
	return Get(Opt_Backend_scheduler_reuse);
}

bool Options::IsBackendSchedule_CBarrier()
{
	return Get(Opt_Backend_scheduler_cbarrier);
}

Options::BackendScheduleHeuristic Options::GetBackendSchedule_Heuristic()
{
	// Opt_Backend_scheduler_function
	return BackendScheduleHeuristic::Default;
}

// Link

bool Options::IsLink_External()
{
	return Get(Opt_Link_external);
}

// Algorithm

Options::ReductionKind Options::GetAlgorithm_ReductionKind()
{
	auto& reductionMode = Get<std::string>(Opt_Algo_reduction);
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

Options::SortKind Options::GetAlgorithm_SortKind()
{
	auto& sortMode = Get<std::string>(Opt_Algo_sort);
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

Options::GroupKind Options::GetAlgorithm_GroupKind()
{
	auto& groupMode = Get<std::string>(Opt_Algo_group);
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

Options::JoinKind Options::GetAlgorithm_JoinKind()
{
	auto& joinMode = Get<std::string>(Opt_Algo_join);
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

unsigned int Options::GetAlgorithm_HashSize()
{
	return Get<unsigned int>(Opt_Algo_hash_size);
}

Options::LikeKind Options::GetAlgorithm_LikeKind()
{
	auto& likeMode = Get<std::string>(Opt_Algo_like);
	if (likeMode == "pcre")
	{
		return LikeKind::PCRELike;
	}
	else if (likeMode == "internal")
	{
		return LikeKind::InternalLike;
	}
	Utils::Logger::LogError("Unknown like mode '" + likeMode + "'");
}

Options::UniqueKind Options::GetAlgorithm_UniqueKind()
{
	auto& uniqueKind = Get<std::string>(Opt_Algo_unique);
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

bool Options::IsData_LoadTPCH()
{
	return Get(Opt_Data_load_tpch);
}

std::string Options::GetData_PathTPCH()
{
	return Get<std::string>(Opt_Data_path_tpch); 
}

float Options::GetData_Resize()
{
	return Get<float>(Opt_Data_resize);
}

Options::AllocatorKind Options::GetData_AllocatorKind()
{
	auto& allocator = Get<std::string>(Opt_Data_allocator);
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

unsigned long long Options::GetData_PageSize()
{
	return Get<unsigned long long>(Opt_Data_page_size);
}

unsigned int Options::GetData_PageCount()
{
	return Get<unsigned int>(Opt_Data_page_count);
}

// Input file

bool Options::HasInputFile()
{
	return Present(Opt_File);
}

std::string Options::GetInputFile()
{
	return Get<std::string>(Opt_File);
}

// Getters

bool Options::Present(const std::string& name)
{
	return GetInstance().m_results.count(name) > 0;
}

bool Options::Get(const std::string& name)
{
	auto& results = GetInstance().m_results;
	// Count does not include default options by default
	try {
		return results[name].as<bool>();
	}
	catch (const cxxopts::option_not_present_exception& ex) {
		return false;
	}
}

template<typename T>
const T& Options::Get(const std::string& name)
{
	return GetInstance().m_results[name].as<T>();
}

// Initialization

Options::Options() : m_options("r3d3", "Optimizing JIT compiler/assembler for HorseIR targetting NVIDIA GPUs")
{
	m_options.add_options()
		("h,help", "Display this help menu")
	;
	m_options.add_options("Optimization")
		(Opt_Optimize_outline, "Outline graph optimization\n"
			"   - none  GPU capable operations executed in isolation\n"
			"   - flow   Merge data-dependent kernels\n"
			"   - full   Optimize shared loads/compression",
			cxxopts::value<std::string>()->default_value("full")
		)
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
		(Opt_Frontend_print_analysis, "Print frontend analyses\n"
			"   - live         Live variables\n"
			"   - rdef         Reaching definitions\n"
			"   - uddu         UD/DU chains\n"
			"   - comp         Compatibility\n"
			"   - dep          Dependency\n"
			"   - shape        Shapes\n"
			"   - geom         Geometry\n"
			"   - kernelgeom   Kernel geometry\n"
			"   - kernelopts   Kernel options\n"
			"   - dataobj      Data object\n"
			"   - datainit     Data initialization",
			cxxopts::value<std::vector<std::string>>()
		)
		(Opt_Frontend_print_analysis_func, "Print frontend analyses for subset of functions", cxxopts::value<std::vector<std::string>>())
		(Opt_Frontend_print_outline, "Pretty print outlined HorseIR program")
		(Opt_Frontend_print_ptx, "Print generated PTX code")
		(Opt_Frontend_print_json, "Print generated PTX JSON")
	;
	m_options.add_options("Backend")
		(Opt_Backend, "Backend assembler\n"
			"   - ptxas      Official NVIDIA assembler\n"
			"   - r3d3       Homegrown assembler",
			cxxopts::value<std::string>()->default_value("r3d3")
		)
		(Opt_Backend_reg_alloc, "Register allocation algorithm\n"
			"   - virtual    Each variable assigned a virtual register\n"
			"   - linear     Linear scan register allocator",
			cxxopts::value<std::string>()->default_value("linear")
		)
		(Opt_Backend_scheduler, "Scheduler algorithm\n"
			"   - linear     Pipelining disabled, instruction order maintained\n"
			"   - list       Pipelining List scheduler (options below)",
			cxxopts::value<std::string>()->default_value("list")
		)
		(Opt_Backend_inline_branch, "Enable inlined (predicated) control-flow branches", cxxopts::value<bool>()->default_value("true"))
		(Opt_Backend_inline_branch_threshold, "Maximum statements in inlined branch", cxxopts::value<unsigned int>()->default_value("6"))
		(Opt_Backend_dump_elf, "Dump assembled .cubin ELF file")
		(Opt_Backend_print_analysis, "Print backend analyses\n"
			"   - live       Live variables\n"
			"   - interval   Live intervals\n"
			"   - rdef       Reaching definitions\n"
			"   - dom        Dominators\n"
			"   - pdom       Post-dominators\n"
			"   - reg        Register allocation\n"
			"   - param      Parameter allocation\n"
			"   - dep        Dependencies",
			cxxopts::value<std::vector<std::string>>()
		)
		(Opt_Backend_print_analysis_func, "Print backend analyses for subset of functions", cxxopts::value<std::vector<std::string>>())
		(Opt_Backend_print_analysis_block, "Print backend analyses in basic blocks mode")
		(Opt_Backend_print_cfg, "Print control-flow graph")
		(Opt_Backend_print_structured, "Print structured control-flow graph")
		(Opt_Backend_print_sass, "Print generated SASS code")
		(Opt_Backend_print_scheduled, "Print scheduled SASS code")
		(Opt_Backend_print_assembled, "Print assembled SASS code")
		(Opt_Backend_print_elf, "Print generated ELF file")
	;
	m_options.add_options("Backend Scheduler")
		(Opt_Backend_scheduler_dual, "Dual issue instructions", cxxopts::value<bool>()->default_value("true"))
		(Opt_Backend_scheduler_reuse, "Enable register reuse flags", cxxopts::value<bool>()->default_value("true"))
		(Opt_Backend_scheduler_cbarrier, "Data dependence counting barriers", cxxopts::value<bool>()->default_value("true"))
		(Opt_Backend_scheduler_function, "Schedule heuristic function")
	;
	m_options.add_options("Link")
		(Opt_Link_external, "Link external libraries (libdevice)")
	;
	m_options.add_options("Algorithm")
		(Opt_Algo_reduction, "Reduction mode\n"
			"   - shflwarp     Warp shuffle reduction\n"
			"   - shflblock    Block shuffle reduction\n"
			"   - shared       Shared memory reduction",
			cxxopts::value<std::string>()->default_value("shflwarp")
		)
		(Opt_Algo_sort, "Sort mode\n"
			"   - global       Global memory bitonic sort\n"
			"   - shared       Shared memory bitonic sort",
			cxxopts::value<std::string>()->default_value("shared")
		)
		(Opt_Algo_group, "Group mode\n"
			"   - cell         Each cell allocated a buffer\n"
			"   - compressed   Cells share a subdivided buffer",
			cxxopts::value<std::string>()->default_value("compressed")
		)
		(Opt_Algo_join, "Join mode\n"
			"   - loop         O(N^2) loop join\n"
			"   - hash         Hash join (murmur3)",
			cxxopts::value<std::string>()->default_value("hash")
		)
		(Opt_Algo_hash_size, "Hash table size [data * 2^n]", cxxopts::value<unsigned int>()->default_value("1"))
		(Opt_Algo_like, "Like mode\n"
			"   - pcre         jpcre2 regex library\n"
			"   - internal     Internal LIKE implementation",
			cxxopts::value<std::string>()->default_value("internal")
		)
		(Opt_Algo_unique, "Unique mode\n"
			"   - sort         Sort based unique\n"
			"   - loop         O(N^2) loop unique",
			cxxopts::value<std::string>()->default_value("loop")
		)
	;
	m_options.add_options("Backend")
	;
	m_options.add_options("Data")
		(Opt_Data_load_tpch, "Load TPC-H data")
		(Opt_Data_path_tpch, "TPC-H data path", cxxopts::value<std::string>())
		(Opt_Data_allocator, "GPU allocator algorithm\n"
			"   - cuda     Default CUDA allocation scheme\n"
			"   - linear   Pre-allocated data pages",
			cxxopts::value<std::string>()->default_value("linear")
		)
		(Opt_Data_page_size, "GPU page size", cxxopts::value<unsigned long long>()->default_value("2147483648"))
		(Opt_Data_page_count, "GPU page count", cxxopts::value<unsigned int>()->default_value("2"))
		(Opt_Data_resize, "Resize buffer factor (only used with cuda allocator)", cxxopts::value<float>()->default_value("0.9"))
	;
	m_options.add_options("Query")
		(Opt_File, "Query HorseIR file", cxxopts::value<std::string>())
	;
	m_options.parse_positional(Opt_File);
	m_options.positional_help("filename");
	m_options.set_width(150);
}

Options& Options::GetInstance()
{
	static Options instance;
	return instance;
}

}
