#pragma once

#include "Libraries/cxxopts.hpp"

namespace Utils {

class Options
{
public:
	static constexpr char const *Opt_Help = "help";
	static constexpr char const *Opt_Dev = "dev";

	static constexpr char const *Opt_Optimize_outline = "optimize-outline";
	static constexpr char const *Opt_Optimize_hir = "optimize-hir";
	static constexpr char const *Opt_Optimize_ptx = "optimize-ptx";
	static constexpr char const *Opt_Optimize_sass = "optimize-sass";
	static constexpr char const *Opt_Optimize_ptxas_level = "optimize-ptxas-level";
	static constexpr char const *Opt_Optimize_ptxas_expensive = "optimize-ptxas-expensive";

	static constexpr char const *Opt_Debug_options = "debug-options";
	static constexpr char const *Opt_Debug_load = "debug-load";
	static constexpr char const *Opt_Debug_print = "debug-print";
	static constexpr char const *Opt_Debug_time = "debug-time";
	static constexpr char const *Opt_Debug_time_unit = "debug-time-unit";
	static constexpr char const *Opt_Debug_compile_only = "debug-compile-only";

	static constexpr char const *Opt_Frontend_print_hir = "frontend-print-hir";
	static constexpr char const *Opt_Frontend_print_hir_typed = "frontend-print-hir-typed";
	static constexpr char const *Opt_Frontend_print_symbols = "frontend-print-symbols";
	static constexpr char const *Opt_Frontend_print_analysis = "frontend-print-analysis";
	static constexpr char const *Opt_Frontend_print_analysis_func = "frontend-print-analysis-func";
	static constexpr char const *Opt_Frontend_print_outline = "frontend-print-outline";
	static constexpr char const *Opt_Frontend_print_ptx = "frontend-print-ptx";
	static constexpr char const *Opt_Frontend_print_json = "frontend-print-json";

	static constexpr char const *Opt_Backend = "backend";
	static constexpr char const *Opt_Backend_reg_alloc = "backend-reg-alloc";
	static constexpr char const *Opt_Backend_inline_branch = "backend-inline-branch";
	static constexpr char const *Opt_Backend_inline_branch_threshold = "backend-inline-branch-threshold";
	static constexpr char const *Opt_Backend_scheduler = "backend-scheduler";
	static constexpr char const *Opt_Backend_load_elf = "backend-load-elf";
	static constexpr char const *Opt_Backend_save_elf = "backend-save-elf";
	static constexpr char const *Opt_Backend_print_analysis = "backend-print-analysis";
	static constexpr char const *Opt_Backend_print_analysis_func = "backend-print-analysis-func";
	static constexpr char const *Opt_Backend_print_analysis_block = "backend-print-analysis-block";
	static constexpr char const *Opt_Backend_print_cfg = "backend-print-cfg";
	static constexpr char const *Opt_Backend_print_structured = "backend-print-structured";
	static constexpr char const *Opt_Backend_print_sass = "backend-print-sass";
	static constexpr char const *Opt_Backend_print_scheduled = "backend-print-scheduled";
	static constexpr char const *Opt_Backend_print_assembled = "backend-print-assembled";
	static constexpr char const *Opt_Backend_print_elf = "backend-print-elf";

	static constexpr char const *Opt_Backend_scheduler_dual = "backend-scheduler-dual";
	static constexpr char const *Opt_Backend_scheduler_reuse = "backend-scheduler-reuse";
	static constexpr char const *Opt_Backend_scheduler_cbarrier = "backend-scheduler-cbarrier";
	static constexpr char const *Opt_Backend_scheduler_function = "backend-scheduler-function";

	static constexpr char const *Opt_Assembler_link_external = "assembler-link-external";
	static constexpr char const *Opt_Assembler_load_elf = "assembler-load-elf";
	static constexpr char const *Opt_Assembler_save_elf = "assembler-save-elf";

	static constexpr char const *Opt_Algo_reduction = "algo-reduction";
	static constexpr char const *Opt_Algo_sort = "algo-sort";
	static constexpr char const *Opt_Algo_group = "algo-group";
	static constexpr char const *Opt_Algo_join = "algo-join";
	static constexpr char const *Opt_Algo_hash_size = "algo-hash-size";
	static constexpr char const *Opt_Algo_like = "algo-like";
	static constexpr char const *Opt_Algo_unique = "algo-unique";
	static constexpr char const *Opt_Algo_member = "algo-member";

	static constexpr char const *Opt_Data_load_tpch = "data-load-tpch";
	static constexpr char const *Opt_Data_resize = "data-resize";
	static constexpr char const *Opt_Data_allocator = "data-allocator";
	static constexpr char const *Opt_Data_page_size = "data-page-size";
	static constexpr char const *Opt_Data_page_count = "data-page-count";

	static constexpr char const *Opt_File = "file";

	Options(Options const&) = delete;
	void operator=(Options const&) = delete;

	static void Initialize(int argc, const char *argv[]);

	static bool IsDev();

	// Optimization

	enum class OutlineOptimization {
		None,
		Flow,
		Full
	};

	static OutlineOptimization GetOptimize_Outline();

	static bool IsOptimize_HorseIR();
	static bool IsOptimize_PTX();
	static bool IsOptimize_SASS();

	static unsigned int GetOptimize_PtxasLevel();
	static bool IsOptimize_PtxasExpensive();

	// Debug

	enum class TimeUnit {
		Microseconds,
		Nanoseconds
	};

	static bool IsDebug_Options();
	static bool IsDebug_Load();
	static bool IsDebug_Print();
	static bool IsDebug_Time();
	static TimeUnit GetDebug_TimeUnit();
	static bool IsDebug_CompileOnly();

	// Frontend

	static bool IsFrontend_PrintHorseIR();
	static bool IsFrontend_PrintHorseIRTyped();
	static bool IsFrontend_PrintSymbols();
	static bool IsFrontend_PrintAnalysis(const std::string& analysis, const std::string& function);
	static bool IsFrontend_PrintOutline();
	static bool IsFrontend_PrintPTX();
	static bool IsFrontend_PrintJSON();

	// Backend

	enum class BackendKind {
		ptxas,
		r4d4
	};

	enum class BackendRegisterAllocator {
		Virtual,
		LinearScan
	};

	enum class BackendScheduler {
		Linear,
		List
	};

	static BackendKind GetBackend_Kind();
	static BackendRegisterAllocator GetBackend_RegisterAllocator();
	static BackendScheduler GetBackend_Scheduler();

	static bool IsBackend_InlineBranch();
	static unsigned int GetBackend_InlineBranchThreshold();

	static bool IsBackend_LoadELF();
	static bool IsBackend_SaveELF();
	static const std::string& GetBackend_LoadELFFile();
	static const std::string& GetBackend_SaveELFFile();

	static bool IsBackend_PrintAnalysis(const std::string& analysis, const std::string& function);
	static bool IsBackend_PrintAnalysisBlock();
	static bool IsBackend_PrintCFG();
	static bool IsBackend_PrintStructured();
	static bool IsBackend_PrintScheduled();
	static bool IsBackend_PrintSASS();
	static bool IsBackend_PrintAssembled();
	static bool IsBackend_PrintELF();

	// Backend scheduler

	static bool IsBackendSchedule_Dual();
	static bool IsBackendSchedule_Reuse();
	static bool IsBackendSchedule_CBarrier();

	enum class BackendScheduleHeuristic {
		Default
	};

	static BackendScheduleHeuristic GetBackendSchedule_Heuristic();

	// Assembler

	static bool IsAssembler_LinkExternal();

	static bool IsAssembler_LoadELF();
	static bool IsAssembler_SaveELF();
	static const std::string& GetAssembler_LoadELFFile();
	static const std::string& GetAssembler_SaveELFFile();

	// Algorithm

	enum class ReductionKind {
		ShuffleBlock,
		ShuffleWarp,
		Shared
	};

	enum class SortKind {
		GlobalSort,
		SharedSort
	};

	enum class GroupKind {
		CellGroup,
		CompressedGroup
	};

	enum class JoinKind {
		LoopJoin,
		HashJoin
	};

	enum class LikeKind {
		PCRELike,
		InternalLike,
		GPULike,
		GPULikeCache
	};

	enum class UniqueKind {
		SortUnique,
		LoopUnique
	};

	enum class MemberKind {
		LoopMember,
		HashMember
	};

	static ReductionKind GetAlgorithm_ReductionKind();
	static SortKind GetAlgorithm_SortKind();
	static GroupKind GetAlgorithm_GroupKind();
	static JoinKind GetAlgorithm_JoinKind();
	static unsigned int GetAlgorithm_HashSize();
	static LikeKind GetAlgorithm_LikeKind();
	static UniqueKind GetAlgorithm_UniqueKind();
	static MemberKind GetAlgorithm_MemberKind();

	// Data

	static bool IsData_LoadTPCH();
	static const std::string& GetData_PathTPCH();

	static float GetData_Resize();

	enum class AllocatorKind {
		CUDA,
		Linear
	};

	static AllocatorKind GetData_AllocatorKind();
	static unsigned long long GetData_PageSize();
	static unsigned int GetData_PageCount();

	// Input file

	static bool HasInputFile();
	static const std::string& GetInputFile();

private:
	// Getter

	static bool Present(const std::string& name);
	static bool Get(const std::string& name);

	template<typename T>
	static const T& Get(const std::string& name);

	// Initialization

	Options();
	static Options& GetInstance();

	cxxopts::Options m_options;
	cxxopts::ParseResult m_results;
};

}
