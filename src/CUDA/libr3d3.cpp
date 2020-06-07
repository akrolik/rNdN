#include "CUDA/libr3d3.h"

namespace CUDA {

PTX::Program *libr3d3::CreateProgram(const std::unique_ptr<Device>& device)
{
	auto program = new PTX::Program();
	auto module = new PTX::Module();
	module->SetVersion(PTX::MAJOR_VERSION, PTX::MINOR_VERSION);
	module->SetDeviceTarget(device->GetComputeCapability());
	module->SetAddressSize(PTX::Bits::Bits64);

	CreateFunction_set<PTX::Bits::Bits64, PTX::Int8Type>(module, "i8");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int16Type>(module, "i16");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int32Type>(module, "i32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Int64Type>(module, "i64");

	CreateFunction_set<PTX::Bits::Bits64, PTX::Float32Type>(module, "f32");
	CreateFunction_set<PTX::Bits::Bits64, PTX::Float64Type>(module, "f64");

	CreateFunction_initlist<PTX::Bits::Bits64>(module);

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
	kernel->AddStatement(addressRegisterDeclaration);

	auto addressRegister0 = addressRegisterDeclaration->GetVariable("rd_a", 0);
	auto addressRegister1 = addressRegisterDeclaration->GetVariable("rd_a", 1);
	auto addressRegister2 = addressRegisterDeclaration->GetVariable("rd_a", 2);
	auto addressRegister3 = addressRegisterDeclaration->GetVariable("rd_a", 3);

	auto valueRegisterDeclaration = new PTX::RegisterDeclaration<T>("v", 1);
	kernel->AddStatement(valueRegisterDeclaration);
	auto valueRegister = valueRegisterDeclaration->GetVariable("v", 0);

	auto indexRegisterDeclaration = new PTX::RegisterDeclaration<PTX::UInt32Type>("r_i", 5);
	kernel->AddStatement(indexRegisterDeclaration);

	auto indexRegister0 = indexRegisterDeclaration->GetVariable("r_i", 0);
	auto indexRegister1 = indexRegisterDeclaration->GetVariable("r_i", 1);
	auto indexRegister2 = indexRegisterDeclaration->GetVariable("r_i", 2);
	auto indexRegister3 = indexRegisterDeclaration->GetVariable("r_i", 3);
	auto indexRegister4 = indexRegisterDeclaration->GetVariable("r_i", 4);

	auto predicateRegisterDeclaration = new PTX::RegisterDeclaration<PTX::PredicateType>("p", 1);
	kernel->AddStatement(predicateRegisterDeclaration);
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

	auto indexed_tid = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
	auto indexed_ntid = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);
	auto indexed_ctaid = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);

	kernel->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(indexRegister1, indexed_tid));
	kernel->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(indexRegister2, indexed_ntid));
	kernel->AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(indexRegister3, indexed_ctaid));

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

	kernel->AddStatement(endLabel);
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
		".reg .pred 	%p<3>;\n\t"
		".reg .b32 	%r<8>;\n\t"
		".reg .b64 	%rd<27>;\n\t"

		"ld.param.u64 	%rd7, [o_adata];\n\t"
		"ld.param.u64 	%rd8, [o_asizes];\n\t"
		"ld.param.u64 	%rd9, [o_sizes];\n\t"
		"ld.param.u64 	%rd10, [i_offsets];\n\t"
		"ld.param.u64 	%rd11, [i_data];\n\t"
		"ld.param.u32 	%r2, [i_size];\n\t"
		"ld.param.u32 	%r3, [i_datasize];\n\t"

		"mov.u32 	%r4, %ntid.x;\n\t"
		"mov.u32 	%r5, %ctaid.x;\n\t"
		"mov.u32 	%r6, %tid.x;\n\t"
		"mad.lo.s32 	%r1, %r4, %r5, %r6;\n\t"
		"setp.ge.u32	%p1, %r1, %r2;\n\t"
		"@%p1 bra 	BB0_5;\n\n\t"

		"cvt.u64.u32	%rd1, %r1;\n\t"
		"cvta.to.global.u64 	%rd12, %rd10;\n\t"
		"mul.wide.u32 	%rd13, %r1, 8;\n\t"
		"add.s64 	%rd2, %rd12, %rd13;\n\t"
		"ld.global.u64 	%rd3, [%rd2];\n\t"
		"add.s32 	%r7, %r1, 1;\n\t"
		"setp.eq.s32	%p2, %r7, %r2;\n\t"
		"@%p2 bra 	BB0_3;\n\t"
		"bra.uni 	BB0_2;\n\n"

	"BB0_3:\n\t"
		"cvt.u64.u32	%rd26, %r3;\n\t"
		"bra.uni 	BB0_4;\n\n"

	"BB0_2:\n\t"
		"ld.global.u64 	%rd26, [%rd2+8];\n\n"

	"BB0_4:\n\t"
		"shl.b64 	%rd14, %rd3, 3;\n\t"
		"add.s64 	%rd15, %rd14, %rd11;\n\t"
		"cvta.to.global.u64 	%rd16, %rd7;\n\t"
		"shl.b64 	%rd17, %rd1, 3;\n\t"
		"add.s64 	%rd18, %rd16, %rd17;\n\t"
		"st.global.u64 	[%rd18], %rd15;\n\t"
		"shl.b64 	%rd19, %rd1, 2;\n\t"
		"add.s64 	%rd20, %rd19, %rd9;\n\t"
		"cvta.to.global.u64 	%rd21, %rd8;\n\t"
		"add.s64 	%rd22, %rd21, %rd17;\n\t"
		"st.global.u64 	[%rd22], %rd20;\n\t"
		"sub.s64 	%rd23, %rd26, %rd3;\n\t"
		"cvta.to.global.u64 	%rd24, %rd9;\n\t"
		"add.s64 	%rd25, %rd24, %rd19;\n\t"
		"st.global.u32 	[%rd25], %rd23;\n\n"

	"BB0_5:\n\t"
		"ret"
	));

	module->AddDeclaration(kernel);
	module->AddEntryFunction(kernel);
}

}
