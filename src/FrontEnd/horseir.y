%{
extern int yylineno;

#include "HorseIR/Tree/Program.h"
#include "Utils/Logger.h"

extern HorseIR::Program *program;

int yylex();
void yyerror(const char *s)
{
	Utils::Logger::LogError("(line " + std::to_string(yylineno) + ") " + s);
}
%}

%code requires {
	#include <string>
	#include <vector>

	#include "HorseIR/Tree/Import.h"
	#include "HorseIR/Tree/Method.h"
	#include "HorseIR/Tree/Module.h"
	#include "HorseIR/Tree/ModuleContent.h"
	#include "HorseIR/Tree/Parameter.h"
	#include "HorseIR/Tree/Program.h"
	#include "HorseIR/Tree/Expressions/CallExpression.h"
	#include "HorseIR/Tree/Expressions/CastExpression.h"
	#include "HorseIR/Tree/Expressions/Expression.h"
	#include "HorseIR/Tree/Expressions/Identifier.h"
	#include "HorseIR/Tree/Expressions/Literal.h"
	#include "HorseIR/Tree/Expressions/Symbol.h"
	#include "HorseIR/Tree/Statements/AssignStatement.h"
	#include "HorseIR/Tree/Statements/ReturnStatement.h"
	#include "HorseIR/Tree/Statements/Statement.h"
	#include "HorseIR/Tree/Types/ListType.h"
	#include "HorseIR/Tree/Types/PrimitiveType.h"
	#include "HorseIR/Tree/Types/Type.h"
}

%locations
%error-verbose

%union {
	long int_val;
	double float_val;
	std::string *string_val;

	std::vector<HorseIR::ModuleContent *> *module_contents;
	std::vector<HorseIR::Parameter *> *parameters;
	std::vector<HorseIR::Statement *> *statements;
	std::vector<HorseIR::Expression *> *expressions;
	std::vector<std::string> *strings;
	std::vector<long> *ints;
	std::vector<double> *floats;

	HorseIR::ModuleContent *module_content;
	HorseIR::Parameter *parameter;
	HorseIR::Type *type;
	HorseIR::Statement *statement;
	HorseIR::Expression *expression;
}

%token '(' ')' '{' '}' '<' '>'
%token tBOOL tI8 tI16 tI32 tI64 tF32 tF64 tCOMPLEX tSYMBOL tSTRING tLIST tTABLE tDATE
%token tMODULE tIMPORT tDEF tCHECKCAST tRETURN
%token <int_val> tINTVAL tDATEVAL
%token <float_val> tFLOATVAL
%token <string_val> tSTRINGVAL
%token <string_val> tFUNCTIONVAL
%token <string_val> tSYMBOLVAL
%token <string_val> tIDENTIFIER

%type <module_contents> module_contents
%type <module_content> module_content
%type <parameters> parametersne parameters
%type <parameter> parameter
%type <type> type int_type float_type
%type <statements> statements
%type <statement> statement
%type <expressions> literals literalsne
%type <expression> expression call literal
%type <ints> int_list date_list
%type <floats> float_list
%type <strings> string_list

%start program

%%

program : tMODULE tIDENTIFIER '{' module_contents '}'                           { program = new HorseIR::Program({new HorseIR::Module(*$2, *$4)}); }
	;

module_contents : module_contents module_content                                { $1->push_back($2); $$ = $1; }
	        | %empty                                                        { $$ = new std::vector<HorseIR::ModuleContent *>(); }
                ;

module_content : tDEF tIDENTIFIER '(' parameters ')' ':' type '{' statements '}'{ $$ = new HorseIR::Method(*$2, *$4, $7, *$9); }
	       | tIMPORT tIDENTIFIER ';'                                        { $$ = new HorseIR::Import(*$2); }
               ;

parameters : parametersne                                                       { $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Parameter *>(); }
           ;

parametersne : parametersne ',' parameter                                       { $1->push_back($3); $$ = $1; }
	     | parameter                                                        { $$ = new std::vector<HorseIR::Parameter *>({$1}); }
             ;

parameter : tIDENTIFIER ':' type                                                { $$ = new HorseIR::Parameter(*$1, $3); }
	  ;

type : '?'                                                                      { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Wildcard); }
     | tBOOL                                                                    { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Bool); }
     | int_type                                                                 { $$ = $1; }
     | float_type                                                               { $$ = $1; }
     | tSTRING                                                                  { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::String); }
     | tSYMBOL                                                                  { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Symbol); }
     | tTABLE                                                                   { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Table); }
     | tDATE                                                                    { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int64); } /* TODO date type */
     | tLIST '<' type '>'                                                       { $$ = new HorseIR::ListType($3); }
     ;

int_type : tI8                                                                  { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int8); }
         | tI16                                                                 { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int16); }
         | tI32                                                                 { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int32); }
         | tI64                                                                 { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int64); }
	 ;

float_type : tF32                                                               { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float32); }
           | tF64                                                               { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float64); }
           ;

statements : statements statement                                               { $1->push_back($2); $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Statement *>(); }
           ;

statement : tIDENTIFIER ':' type '=' expression ';'                             { $$ = new HorseIR::AssignStatement(*$1, $3, $5); }
          | tRETURN tIDENTIFIER ';'                                             { $$ = new HorseIR::ReturnStatement(new HorseIR::Identifier(*$2)); }
          ;

expression : tCHECKCAST '(' expression ',' type ')'                             { $$ = new HorseIR::CastExpression($3, $5); }
	   | call                                                               { $$ = $1; }
           | literal                                                            { $$ = $1; }
           ; 

call : tFUNCTIONVAL '(' literals ')'                                            { $$ = new HorseIR::CallExpression(*$1, *$3); }
     ;

literals : literalsne                                                           { $$ = $1; }
         | %empty                                                               { $$ = new std::vector<HorseIR::Expression *>(); }
         ;

literalsne : literalsne ',' literal                                             { $1->push_back($3); $$ = $1; }
           | literal                                                            { $$ = new std::vector<HorseIR::Expression *>({$1}); } 
           ;

literal : tIDENTIFIER                                                           { $$ = new HorseIR::Identifier(*$1); }
	| tSYMBOLVAL ':' tSYMBOL                                                { $$ = new HorseIR::Symbol(*$1); }
	| int_list ':' int_type                                                 { $$ = new HorseIR::Literal<int64_t>(*$1, $3); }
        | '(' int_list ')' ':' int_type                                         { $$ = new HorseIR::Literal<int64_t>(*$2, $5); }
	| int_list ':' float_type                                               { $$ = new HorseIR::Literal<int64_t>(*$1, $3); }
        | '(' int_list ')' ':' float_type                                       { $$ = new HorseIR::Literal<int64_t>(*$2, $5); }
	| float_list ':' float_type                                             { $$ = new HorseIR::Literal<double>(*$1, $3); }
        | '(' float_list ')' ':' float_type                                     { $$ = new HorseIR::Literal<double>(*$2, $5); }
        | string_list ':' tSTRING                                               { $$ = new HorseIR::Literal<std::string>(*$1, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::String)); }
        | '(' string_list ')' ':' tSTRING                                       { $$ = new HorseIR::Literal<std::string>(*$2, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::String)); }
        | date_list ':' tDATE                                                   { $$ = new HorseIR::Literal<int64_t>(*$1, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int64)); }
        | '(' date_list ')' ':' tDATE                                           { $$ = new HorseIR::Literal<int64_t>(*$2, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int64)); }
        ;

int_list : int_list ',' tINTVAL                                                 { $1->push_back($3); $$ = $1; } 
	 | tINTVAL                                                              { $$ = new std::vector<long>({$1}); }
         ;

float_list : float_list ',' tFLOATVAL                                           { $1->push_back($3); $$ = $1; } 
           | tFLOATVAL                                                          { $$ = new std::vector<double>({$1}); }
           ;

string_list : string_list ',' tSTRINGVAL                                        { $1->push_back(*$3); $$ = $1; } 
	    | tSTRINGVAL                                                        { $$ = new std::vector<std::string>({*$1}); }
            ;

date_list : date_list ',' tDATEVAL                                              { $1->push_back($3); $$ = $1; } 
	  | tDATEVAL                                                            { $$ = new std::vector<long>({$1}); }
          ;

%%
