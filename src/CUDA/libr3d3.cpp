#include "CUDA/libr3d3.h"

namespace CUDA {

PTX::Program *libr3d3::CreateProgram(const std::unique_ptr<Device>& device)
{
	auto program = new PTX::Program();
	auto module = new PTX::Module();
	module->SetVersion(PTX::MAJOR_VERSION, PTX::MINOR_VERSION);
	module->SetDeviceTarget(device->GetComputeCapability());
	module->SetAddressSize(PTX::Bits::Bits64);

	CreateFunction_set<PTX::Bits::Bits64, PTX::UInt8Type>(module, "u8");
	CreateFunction_set<PTX::Bits::Bits64, PTX::UInt16Type>(module, "u16");
	CreateFunction_set<PTX::Bits::Bits64, PTX::UInt32Type>(module, "u32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::UInt64Type>(module, "u64");

	CreateFunction_set<PTX::Bits::Bits64, PTX::Int8Type>(module, "s8");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int16Type>(module, "s16");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int32Type>(module, "s32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int64Type>(module, "s64");

	CreateFunction_set<PTX::Bits::Bits64, PTX::Float32Type>(module, "f32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Float64Type>(module, "f64");

	CreateFunction_initlist<PTX::Bits::Bits64>(module);

	CreateFunction_like_internal<PTX::Bits::Bits64>(module);
	CreateFunction_like<PTX::Bits::Bits64>(module);
	CreateFunction_like_cache<PTX::Bits::Bits64>(module);

	program->AddModule(module);
	return program;
}

template<PTX::Bits B, class T>
void libr3d3::CreateFunction_set(PTX::Module *module, const std::string& typeName)
{
	// .visible .entry set_i32(
	// 	.param .u64 data,
	// 	.param .u32 value,
	// 	.param .u32 size,
	// )
	// {
	// 	.reg .pred      %p<2>;
	// 	.reg .b32       %r<7>;
	// 	.reg .b64       %rd<5>;

	// 	ld.param.u64    %rd1, [data];
	// 	ld.param.u32    %r2, [value];
	// 	ld.param.u32    %r3, [size];

	// 	mov.u32         %r4, %ctaid.x;
	// 	mov.u32         %r5, %ntid.x;
	// 	mov.u32         %r6, %tid.x;
	// 	mad.lo.u32      %r1, %r5, %r4, %r6;
	// 	setp.ge.u32     %p1, %r1, %r3;
	// 	@%p1 bra        BB0_2;

	// 	cvta.to.global.u64      %rd2, %rd1;
	// 	mul.wide.s32    %rd3, %r1, 4;
	// 	add.s64         %rd4, %rd2, %rd3;
	// 	st.global.s32   [%rd4], %r2;

	// BB0_2:
	// 	ret;
	// }

	// Setup kernel

	auto kernel = new PTX::FunctionDefinition<PTX::VoidType>();
	kernel->SetName("set_" + typeName);
	kernel->SetEntry(true);
	kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

	// Initialize parameters

	auto dataDeclaration = new PTX::PointerDeclaration<B, T>("data");
	auto valueDeclaration = new PTX::ParameterDeclaration<T>("value");
	auto sizeDeclaration = new PTX::ParameterDeclaration<PTX::UInt32Type>("size");

	kernel->AddParameter(dataDeclaration);
	kernel->AddParameter(valueDeclaration);
	kernel->AddParameter(sizeDeclaration);
	
	auto data = dataDeclaration->GetVariable("data");
	auto value = valueDeclaration->GetVariable("value");
	auto size = sizeDeclaration->GetVariable("size");

	// Allocate registers

	auto addressRegisterDeclaration = new PTX::RegisterDeclaration<PTX::UIntType<B>>("rd_a", 4);
	kernel->AddStatement(new PTX::DeclarationStatement(addressRegisterDeclaration));

	auto addressRegister0 = addressRegisterDeclaration->GetVariable("rd_a", 0);
	auto addressRegister1 = addressRegisterDeclaration->GetVariable("rd_a", 1);
	auto addressRegister2 = addressRegisterDeclaration->GetVariable("rd_a", 2);
	auto addressRegister3 = addressRegisterDeclaration->GetVariable("rd_a", 3);

	auto valueRegisterDeclaration = new PTX::RegisterDeclaration<T>("v", 1);
	kernel->AddStatement(new PTX::DeclarationStatement(valueRegisterDeclaration));
	auto valueRegister = valueRegisterDeclaration->GetVariable("v", 0);

	auto indexRegisterDeclaration = new PTX::RegisterDeclaration<PTX::UInt32Type>("r_i", 5);
	kernel->AddStatement(new PTX::DeclarationStatement(indexRegisterDeclaration));

	auto indexRegister0 = indexRegisterDeclaration->GetVariable("r_i", 0);
	auto indexRegister1 = indexRegisterDeclaration->GetVariable("r_i", 1);
	auto indexRegister2 = indexRegisterDeclaration->GetVariable("r_i", 2);
	auto indexRegister3 = indexRegisterDeclaration->GetVariable("r_i", 3);
	auto indexRegister4 = indexRegisterDeclaration->GetVariable("r_i", 4);

	auto predicateRegisterDeclaration = new PTX::RegisterDeclaration<PTX::PredicateType>("p", 1);
	kernel->AddStatement(new PTX::DeclarationStatement(predicateRegisterDeclaration));
	auto predicateRegister = predicateRegisterDeclaration->GetVariable("p", 0);

	// Load parameters

	auto dataAddressBase = new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(data);
	auto dataBase = new PTX::PointerRegisterAdapter<B, T>(addressRegister0);

	auto valueAddress = new PTX::MemoryAddress<B, T, PTX::ParameterSpace>(value);
	auto sizeAddress = new PTX::MemoryAddress<B, PTX::UInt32Type, PTX::ParameterSpace>(size);

	kernel->AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(dataBase, dataAddressBase));
	kernel->AddStatement(new PTX::LoadInstruction<B, T, PTX::ParameterSpace>(valueRegister, valueAddress));
	kernel->AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::ParameterSpace>(indexRegister0, sizeAddress));
	
	// Compute index

	auto indexed_tid = new PTX::IndexedSpecialRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
	auto indexed_ntid = new PTX::IndexedSpecialRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);
	auto indexed_ctaid = new PTX::IndexedSpecialRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);

	kernel->AddStatement(new PTX::MoveSpecialInstruction<PTX::UInt32Type>(indexRegister1, indexed_tid));
	kernel->AddStatement(new PTX::MoveSpecialInstruction<PTX::UInt32Type>(indexRegister2, indexed_ntid));
	kernel->AddStatement(new PTX::MoveSpecialInstruction<PTX::UInt32Type>(indexRegister3, indexed_ctaid));

	kernel->AddStatement(new PTX::MADInstruction<PTX::UInt32Type>(
		indexRegister4, indexRegister2, indexRegister3, indexRegister1, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
	));

	// Bounds check

	auto endLabel = new PTX::Label("END");
	kernel->AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
		predicateRegister, indexRegister4, indexRegister0, PTX::UInt32Type::ComparisonOperator::GreaterEqual
	));
	kernel->AddStatement(new PTX::BranchInstruction(endLabel, predicateRegister));

	// Store result at index

	auto globalBase = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(addressRegister1);
	auto genericAddress = new PTX::RegisterAddress<B, T>(dataBase);

	kernel->AddStatement(new PTX::ConvertToAddressInstruction<B, T, PTX::GlobalSpace>(globalBase, genericAddress));

	if constexpr(B == PTX::Bits::Bits32)
	{
		kernel->AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(addressRegister2),
			new PTX::Bit32Adapter<PTX::UIntType>(indexRegister4),
			new PTX::UInt32Value(std::log2(PTX::BitSize<T::TypeBits>::NumBytes))
		));
	}
	else
	{
		kernel->AddStatement(new PTX::MultiplyWideInstruction<PTX::UInt32Type>(
			addressRegister2, indexRegister4, new PTX::UInt32Value(PTX::BitSize<T::TypeBits>::NumBytes)
		));
	}

	kernel->AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(addressRegister3, addressRegister1, addressRegister2));

	auto dataAddressIndexed = new PTX::RegisterAddress<B, T, PTX::GlobalSpace>(new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(addressRegister3));
	kernel->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(dataAddressIndexed, valueRegister));

	// End kernel with return

	kernel->AddStatement(new PTX::LabelStatement(endLabel));
	kernel->AddStatement(new PTX::ReturnInstruction());
  
	module->AddDeclaration(kernel);
	module->AddEntryFunction(kernel);
}

template<PTX::Bits B>
void libr3d3::CreateFunction_initlist(PTX::Module *module)
{
	// .visible .entry initlist(
	// 	.param .u64 o_adata,
	// 	.param .u64 o_asizes,
	// 	.param .u64 o_sizes,
	// 	.param .u64 i_data,
	// 	.param .u64 i_offsets,
	// 	.param .u32 i_size,
	// 	.param .u32 i_datasize
	// )
	// {
	// 	.reg .pred 	%p<3>;
	// 	.reg .b32 	%r<7>;
	// 	.reg .b64 	%rd<27>;

	// 	ld.param.u64 	%rd7, [o_adata];
	// 	ld.param.u64 	%rd8, [o_asizes];
	// 	ld.param.u64 	%rd9, [o_sizes];
	// 	ld.param.u64 	%rd10, [i_data];
	// 	ld.param.u64 	%rd11, [i_offsets];
	// 	ld.param.u32 	%r2, [i_size];
	// 	ld.param.u32 	%r3, [i_datasize];

	// 	mov.u32 	%r4, %ntid.x;
	// 	mov.u32 	%r5, %ctaid.x;
	// 	mov.u32 	%r6, %tid.x;
	// 	mad.lo.s32 	%r1, %r4, %r5, %r6;
	// 	setp.ge.u32	%p1, %r1, %r2;
	// 	@%p1 bra 	BB0_5;

	// 	cvt.u64.u32	%rd1, %r1;
	// 	cvta.to.global.u64 	%rd12, %rd10;
	// 	mul.wide.u32 	%rd13, %r1, 8;
	// 	add.s64 	%rd2, %rd12, %rd13;
	// 	ld.global.u64 	%rd3, [%rd2];
	// 	add.s32 	%r6, %r1, 1;
	// 	setp.eq.s32	%p2, %r7, %r2;
	// 	@%p2 bra 	BB0_3;
	// 	bra.uni 	BB0_2;

	// BB0_3:
	// 	cvt.u64.u32	%rd26, %r3;
	// 	bra.uni 	BB0_4;

	// BB0_2:
	// 	ld.global.u64 	%rd26, [%rd2+8];

	// BB0_4:
	// 	shl.b64 	%rd14, %rd3, 3;
	// 	add.s64 	%rd15, %rd14, %rd11;
	// 	cvta.to.global.u64 	%rd16, %rd7;
	// 	shl.b64 	%rd17, %rd1, 3;
	// 	add.s64 	%rd18, %rd16, %rd17;
	// 	st.global.u64 	[%rd18], %rd15;

	// 	shl.b64 	%rd19, %rd1, 2;
	// 	add.s64 	%rd20, %rd19, %rd9;
	// 	cvta.to.global.u64 	%rd21, %rd8;
	// 	add.s64 	%rd22, %rd21, %rd17;
	// 	st.global.u64 	[%rd22], %rd20;

	// 	sub.s64 	%rd23, %rd26, %rd3;
	// 	cvta.to.global.u64 	%rd24, %rd9;
	// 	add.s64 	%rd25, %rd24, %rd19;
	// 	st.global.u32 	[%rd25], %rd23;

	// BB0_5:
	// 	ret;
	// }

	// Setup kernel

	auto kernel = new PTX::FunctionDefinition<PTX::VoidType>();
	kernel->SetName("init_list");
	kernel->SetEntry(true);
	kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

	// Initialize parameters

	auto o_adataDeclaration = new PTX::PointerDeclaration<B, PTX::UInt64Type>("o_adata");
	auto o_asizesDeclaration = new PTX::PointerDeclaration<B, PTX::UInt64Type>("o_asizes");
	auto o_sizesDeclaration = new PTX::PointerDeclaration<B, PTX::Int32Type>("o_sizes");

	auto i_offsetsDeclaration = new PTX::PointerDeclaration<B, PTX::Int64Type>("i_offsets");
	auto i_dataDeclaration = new PTX::PointerDeclaration<B, PTX::Int64Type>("i_data");
	auto i_sizeDeclaration = new PTX::ParameterDeclaration<PTX::UInt32Type>("i_size");
	auto i_datasizeDeclaration = new PTX::ParameterDeclaration<PTX::UInt32Type>("i_datasize");

	kernel->AddParameter(o_adataDeclaration);
	kernel->AddParameter(o_asizesDeclaration);
	kernel->AddParameter(o_sizesDeclaration);

	kernel->AddParameter(i_offsetsDeclaration);
	kernel->AddParameter(i_dataDeclaration);
	kernel->AddParameter(i_sizeDeclaration);
	kernel->AddParameter(i_datasizeDeclaration);

	kernel->AddStatement(new PTX::DevInstruction(
		".reg .pred 	%p<3>;\n        "
		".reg .b32 	%r<8>;\n        "
		".reg .b64 	%rd<27>;\n        "

		"ld.param.u64 	%rd7, [o_adata];\n        "
		"ld.param.u64 	%rd8, [o_asizes];\n        "
		"ld.param.u64 	%rd9, [o_sizes];\n        "
		"ld.param.u64 	%rd10, [i_offsets];\n        "
		"ld.param.u64 	%rd11, [i_data];\n        "
		"ld.param.u32 	%r2, [i_size];\n        "
		"ld.param.u32 	%r3, [i_datasize];\n        "

		"mov.u32 	%r4, %ntid.x;\n        "
		"mov.u32 	%r5, %ctaid.x;\n        "
		"mov.u32 	%r6, %tid.x;\n        "
		"mad.lo.s32 	%r1, %r4, %r5, %r6;\n        "
		"setp.ge.u32	%p1, %r1, %r2;\n   "
		"@%p1 bra 	BB0_5;\n\n        "

		"cvt.u64.u32	%rd1, %r1;\n        "
		"cvta.to.global.u64 	%rd12, %rd10;\n        "
		"mul.wide.u32 	%rd13, %r1, 8;\n        "
		"add.s64 	%rd2, %rd12, %rd13;\n        "
		"ld.global.u64 	%rd3, [%rd2];\n        "
		"add.s32 	%r7, %r1, 1;\n        "
		"setp.eq.s32	%p2, %r7, %r2;\n   "
		"@%p2 bra 	BB0_3;\n\n        "
		"bra.uni 	BB0_2;\n\n"

	"BB0_3:\n        "
		"cvt.u64.u32	%rd26, %r3;\n        "
		"bra.uni 	BB0_4;\n\n"

	"BB0_2:\n        "
		"ld.global.u64 	%rd26, [%rd2+8];\n\n"

	"BB0_4:\n        "
		"shl.b64 	%rd14, %rd3, 3;\n        "
		"add.s64 	%rd15, %rd14, %rd11;\n        "
		"cvta.to.global.u64 	%rd16, %rd7;\n        "
		"shl.b64 	%rd17, %rd1, 3;\n        "
		"add.s64 	%rd18, %rd16, %rd17;\n        "
		"st.global.u64 	[%rd18], %rd15;\n        "
		"shl.b64 	%rd19, %rd1, 2;\n        "
		"add.s64 	%rd20, %rd19, %rd9;\n        "
		"cvta.to.global.u64 	%rd21, %rd8;\n        "
		"add.s64 	%rd22, %rd21, %rd17;\n        "
		"st.global.u64 	[%rd22], %rd20;\n        "
		"sub.s64 	%rd23, %rd26, %rd3;\n        "
		"cvta.to.global.u64 	%rd24, %rd9;\n        "
		"add.s64 	%rd25, %rd24, %rd19;\n        "
		"st.global.u32 	[%rd25], %rd23;\n\n"

	"BB0_5:\n        "
		"ret"
	));

	module->AddDeclaration(kernel);
	module->AddEntryFunction(kernel);
}

template<PTX::Bits B>
void libr3d3::CreateFunction_like_internal(PTX::Module *module)
{
	// Setup kernel

	auto kernel = new PTX::FunctionDefinition<PTX::ParameterVariable<PTX::Bit32Type>>();
	kernel->SetName("like_internal");

	// Initialize parameters

	auto i_dataDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("i_data");
	auto i_patternDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("i_pattern");

	kernel->AddParameter(i_dataDeclaration);
	kernel->AddParameter(i_patternDeclaration);

	auto returnDeclaration = new PTX::ParameterDeclaration<PTX::Bit32Type>("return_m");
	kernel->SetReturnDeclaration(returnDeclaration);

	kernel->AddStatement(new PTX::DevInstruction(
		".reg .pred %p<15>;\n        "
		".reg .b16 %rs<28>;\n        "
		".reg .b32 %r<4>;\n        "
		".reg .b64 %rd<22>;\n        "

		"ld.param.u64 %rd1, [i_data];\n        "
		"ld.param.u64 %rd2, [i_pattern];\n        "
		"ld.u8 %rs25, [%rd2];\n        "
		"setp.eq.s16     %p1, %rs25, 0;\n   "
		"@%p1 bra BB0_6;\n\n"

	"BB0_1:\n        "
		"mov.u16 %rs27, 0;\n        "
		"add.s16 %rs11, %rs25, -91;\n        "
		"and.b16 %rs12, %rs11, 255;\n        "
		"setp.gt.u16     %p2, %rs12, 4;\n        "
		"cvt.u64.u16     %rd13, %rs11;\n        "
		"and.b64 %rd12, %rd13, 255;\n   "
		"@%p2 bra BB0_3;\n\n        "
		"bra.uni BB0_2;\n\n"

	"BB0_3:\n        "
		"setp.eq.s16     %p4, %rs25, 37;\n        "
		"add.s64 %rd4, %rd2, 1;\n   "
		"@%p4 bra BB0_7;\n\n        "

		"ld.u8 %rs15, [%rd1];\n        "
		"setp.eq.s16     %p5, %rs15, 0;\n        "
		"setp.ne.s16     %p6, %rs15, %rs25;\n        "
		"or.pred %p7, %p5, %p6;\n   "
		"@%p7 bra BB0_15;\n\n        "

		"add.s64 %rd1, %rd1, 1;\n        "
		"ld.u8 %rs25, [%rd2+1];\n        "
		"setp.eq.s16     %p8, %rs25, 0;\n        "
		"mov.u64 %rd2, %rd4;\n   "
		"@%p8 bra BB0_6;\n\n        "
		"bra.uni BB0_1;\n\n"

	"BB0_2:\n        "
		"cvt.u32.u64     %r1, %rd12;\n        "
		"mov.u64 %rd14, 1;\n        "
		"shl.b64 %rd15, %rd14, %r1;\n        "
		"and.b64 %rd16, %rd15, 19;\n        "
		"setp.ne.s64     %p3, %rd16, 0;\n   "
		"@%p3 bra BB0_15;\n\n        "
		"bra.uni BB0_3;\n\n"

	"BB0_6:\n        "
		"ld.u8 %rs24, [%rd1];\n        "
		"setp.eq.s16     %p14, %rs24, 0;\n        "
		"selp.u16        %rs27, 1, 0, %p14;\n\n"

	"BB0_15:\n        "
		"cvt.u32.u16     %r3, %rs27;\n        "
		"st.param.b32    [return_m+0], %r3;\n        "
		"ret;\n\n"

	"BB0_7:\n        "
		"ld.u8 %rs4, [%rd2+1];\n        "
		"setp.eq.s16     %p9, %rs4, 0;\n        "
		"mov.u16 %rs17, 1;\n   "
		"@%p9 bra BB0_8;\n\n        "

		"ld.u8 %rs26, [%rd1];\n        "
		"setp.eq.s16     %p10, %rs26, 0;\n  "
		"@%p10 bra BB0_15;\n\n        "

		"add.s64 %rd6, %rd2, 2;\n\n"

	"BB0_11:\n        "
		"setp.ne.s16     %p11, %rs26, %rs4;\n  "
		"@%p11 bra BB0_14;\n\n        "

		"add.s64 %rd17, %rd1, 1;\n        "

		"{\n                "
			".param .b64 param0;\n                "
			"st.param.b64    [param0+0], %rd17;\n                "
			".param .b64 param1;\n                "
			"st.param.b64    [param1+0], %rd6;\n                "
			".param .b32 retval0;\n                "
			"call.uni (retval0), like_internal, (param0, param1);\n                "
			"ld.param.b32    %r2, [retval0+0];\n        "
		"}\n        "

		"cvt.u16.u32     %rs21, %r2;\n        "
		"and.b16 %rs22, %rs21, 255;\n        "
		"setp.ne.s16     %p12, %rs22, 0;\n  "
		"@%p12 bra BB0_13;\n\n"

	"BB0_14:\n        "
		"add.s64 %rd8, %rd1, 1;\n        "
		"ld.u8 %rs26, [%rd1+1];\n        "
		"setp.eq.s16     %p13, %rs26, 0;\n        "
		"mov.u64 %rd1, %rd8;\n  "
		"@%p13 bra BB0_15;\n\n        "
		"bra.uni BB0_11;\n\n"

	"BB0_8:\n        "
		"mov.u16 %rs27, %rs17;\n        "
		"bra.uni BB0_15;\n\n"

	"BB0_13:\n        "
		"mov.u16 %rs27, %rs17;\n        "
		"bra.uni BB0_15"
	));

	module->AddDeclaration(kernel);
}

template<PTX::Bits B>
void libr3d3::CreateFunction_like(PTX::Module *module)
{
	// Setup kernel

	auto kernel = new PTX::FunctionDefinition<PTX::VoidType>();
	kernel->SetName("like");
	kernel->SetEntry(true);
	kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

	// Initialize parameters

	auto o_matchDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("o_match");
	auto i_indexesDeclaration = new PTX::PointerDeclaration<B, PTX::Int64Type>("i_indexes");
	auto i_dataDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("i_data");
	auto i_patternDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("i_pattern");
	auto i_sizeDeclaration = new PTX::ParameterDeclaration<PTX::UInt32Type>("i_size");

	kernel->AddParameter(o_matchDeclaration);
	kernel->AddParameter(i_indexesDeclaration);
	kernel->AddParameter(i_dataDeclaration);
	kernel->AddParameter(i_patternDeclaration);
	kernel->AddParameter(i_sizeDeclaration);

	kernel->AddStatement(new PTX::DevInstruction(
		".reg .pred %p<2>;\n        "
		".reg .b32 %r<7>;\n        "
		".reg .b64 %rd<13>;\n        "

		"ld.param.u64 %rd1, [o_match];\n        "
		"ld.param.u64 %rd2, [i_indexes];\n        "
		"ld.param.u64 %rd3, [i_data];\n        "
		"ld.param.u64 %rd4, [i_pattern];\n        "
		"ld.param.u32 %r2, [i_size];\n        "
		"mov.u32 %r3, %ctaid.x;\n        "
		"mov.u32 %r4, %ntid.x;\n        "
		"mov.u32 %r5, %tid.x;\n        "
		"mad.lo.s32 %r1, %r4, %r3, %r5;\n        "
		"setp.ge.u32     %p1, %r1, %r2;\n   "
		"@%p1 bra BB1_2;\n\n        "

		"cvta.to.global.u64 %rd5, %rd2;\n        "
		"cvt.u64.u32     %rd6, %r1;\n        "
		"mul.wide.u32 %rd7, %r1, 8;\n        "
		"add.s64 %rd8, %rd5, %rd7;\n        "
		"ld.global.u64 %rd9, [%rd8];\n        "
		"add.s64 %rd10, %rd3, %rd9;\n        "

		"{\n                "
			".param .b64 param0;\n                "
			"st.param.b64    [param0+0], %rd10;\n                "
			".param .b64 param1;\n                "
			"st.param.b64    [param1+0], %rd4;\n                "
			".param .b32 retval0;\n                "
			"call.uni (retval0), like_internal, (param0, param1);\n                "
			"ld.param.b32    %r6, [retval0+0];\n        "
		"}\n        "

		"cvta.to.global.u64 %rd11, %rd1;\n        "
		"add.s64 %rd12, %rd11, %rd6;\n        "
		"st.global.u8 [%rd12], %r6;\n\n"

	"BB1_2:\n        "
		"ret"
	));

	module->AddDeclaration(kernel);
	module->AddEntryFunction(kernel);
}

template<PTX::Bits B>
void libr3d3::CreateFunction_like_cache(PTX::Module *module)
{
	// Setup kernel

	auto kernel = new PTX::FunctionDefinition<PTX::VoidType>();
	kernel->SetName("like_cache");
	kernel->SetEntry(true);
	kernel->SetLinkDirective(PTX::Declaration::LinkDirective::Visible);

	// Initialize parameters

	auto o_cacheDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("o_cache");
	auto i_indexesDeclaration = new PTX::PointerDeclaration<B, PTX::Int64Type>("i_indexes");
	auto i_dataDeclaration = new PTX::PointerDeclaration<B, PTX::Int8Type>("i_data");
	auto i_sizeDeclaration = new PTX::ParameterDeclaration<PTX::UInt32Type>("i_size");

	kernel->AddParameter(o_cacheDeclaration);
	kernel->AddParameter(i_indexesDeclaration);
	kernel->AddParameter(i_dataDeclaration);
	kernel->AddParameter(i_sizeDeclaration);

	kernel->AddStatement(new PTX::DevInstruction(
		".reg .pred %p<17>;\n        "
		".reg .b16 %rs<2>;\n        "
		".reg .b32 %r<15>;\n        "
		".reg .b64 %rd<33>;\n        "

		"ld.param.u64 %rd14, [o_cache];\n        "
		"ld.param.u64 %rd15, [i_indexes];\n        "
		"ld.param.u64 %rd16, [i_data];\n        "
		"ld.param.u32 %r2, [i_size];\n        "
		"mov.u32 %r3, %ntid.x;\n        "
		"mov.u32 %r4, %ctaid.x;\n        "
		"mov.u32 %r5, %tid.x;\n        "
		"mad.lo.s32 %r1, %r3, %r4, %r5;\n        "
		"setp.ge.u32     %p1, %r1, %r2;\n   "
		"@%p1 bra BB2_7;\n\n        "

		"cvta.to.global.u64 %rd17, %rd14;\n        "
		"cvta.to.global.u64 %rd18, %rd15;\n        "
		"mul.wide.u32 %rd19, %r1, 8;\n        "
		"add.s64 %rd20, %rd18, %rd19;\n        "
		"ld.global.u64 %rd1, [%rd20];\n        "
		"add.s64 %rd31, %rd17, %rd1;\n        "
		"ld.global.u8 %rs1, [%rd31];\n        "
		"setp.ne.s16     %p2, %rs1, 0;\n   "
		"@%p2 bra BB2_7;\n\n        "

		"cvta.to.global.u64 %rd21, %rd16;\n        "
		"add.s64 %rd32, %rd21, %rd1;\n        "
		"ld.global.u64 %rd30, [%rd32];\n        "
		"and.b64 %rd22, %rd30, 255;\n        "
		"setp.eq.s64     %p3, %rd22, 0;\n   "
		"@%p3 bra BB2_7;\n\n"

	"BB2_3:\n        "
		"shr.u64 %rd23, %rd30, 8;\n        "
		"shl.b64 %rd24, %rd23, 24;\n        "
		"cvt.u32.u64     %r6, %rd24;\n        "
		"setp.eq.s32     %p4, %r6, 0;\n        "
		"shl.b64 %rd25, %rd30, 8;\n        "
		"cvt.u32.u64     %r7, %rd25;\n        "
		"and.b32 %r8, %r7, -16777216;\n        "
		"setp.eq.s32     %p5, %r8, 0;\n        "
		"or.pred %p6, %p4, %p5;\n        "
		"and.b64 %rd26, %rd30, 4278190080;\n        "
		"setp.eq.s64     %p7, %rd26, 0;\n        "
		"or.pred %p8, %p6, %p7;\n        "
		"cvt.u32.u64     %r9, %rd23;\n        "
		"and.b32 %r10, %r9, -16777216;\n        "
		"setp.eq.s32     %p9, %r10, 0;\n        "
		"or.pred %p10, %p8, %p9;\n  "
		"@%p10 bra BB2_6;\n\n        "

		"st.global.u64 [%rd31], %rd30;\n        "
		"shr.u64 %rd27, %rd30, 16;\n        "
		"cvt.u32.u64     %r11, %rd27;\n        "
		"and.b32 %r12, %r11, -16777216;\n        "
		"setp.eq.s32     %p11, %r12, 0;\n        "
		"shr.u64 %rd28, %rd30, 24;\n        "
		"cvt.u32.u64     %r13, %rd28;\n        "
		"and.b32 %r14, %r13, -16777216;\n        "
		"setp.eq.s32     %p12, %r14, 0;\n        "
		"or.pred %p13, %p11, %p12;\n        "
		"setp.lt.u64     %p14, %rd30, 72057594037927936;\n        "
		"or.pred %p15, %p13, %p14;\n  "
		"@%p15 bra BB2_7;\n\n        "

		"add.s64 %rd31, %rd31, 8;\n        "
		"add.s64 %rd12, %rd32, 8;\n        "
		"ld.global.u64 %rd30, [%rd32+8];\n        "
		"and.b64 %rd29, %rd30, 255;\n        "
		"setp.eq.s64     %p16, %rd29, 0;\n        "
		"mov.u64 %rd32, %rd12;\n  "
		"@%p16 bra BB2_7;\n\n        "
		"bra.uni BB2_3;\n\n"

	"BB2_6:\n        "
		"st.global.u32 [%rd31], %rd30;\n\n"

	"BB2_7:\n        "
		"ret"
	));

	module->AddDeclaration(kernel);
	module->AddEntryFunction(kernel);
}

}
