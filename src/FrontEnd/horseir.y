%{
//TODO: remove these
#include <stdio.h>
#include <stdlib.h>

extern char *yytext;
extern int yylineno;

#include "HorseIR/Tree/Program.h"

extern HorseIR::Program *program;

int yylex();
void yyerror(const char *s) { fprintf(stderr, "[Error] (line %d) %s\n", yylineno, s); exit(1); }
%}

%code requires {
	#include <string>
	#include <vector>

	#include "HorseIR/Tree/Import.h"
	#include "HorseIR/Tree/Method.h"
	#include "HorseIR/Tree/Module.h"
	#include "HorseIR/Tree/ModuleContent.h"
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
	#include "HorseIR/Tree/Types/TableType.h"
	#include "HorseIR/Tree/Types/Type.h"
	#include "HorseIR/Tree/Types/WildcardType.h"
}

%locations
%error-verbose

%union {
	//TODO: use std::string and remove conversions below
	char *string_val;
	long int_val;

	std::vector<HorseIR::ModuleContent *> *module_contents;
	std::vector<HorseIR::Statement *> *statements;
	std::vector<HorseIR::Expression *> *expressions;
	std::vector<std::string> *strings;
	std::vector<long> *ints;

	HorseIR::ModuleContent *module_content;
	HorseIR::Type *type;
	HorseIR::Statement *statement;
	HorseIR::Expression *expression;
}

%token '(' ')' '{' '}' '<' '>'
%token tI64 tSTRING tLIST tTABLE
%token tMODULE tIMPORT tDEF tCHECKCAST tRETURN
%token <int_val> tINTVAL
%token <string_val> tSTRINGVAL
%token <string_val> tFUNCTION
%token <string_val> tSYMBOL
%token <string_val> tIDENTIFIER

%type <module_contents> module_contents
%type <module_content> module_content
%type <type> type int_type
%type <statements> statements
%type <statement> statement
%type <expressions> literals literalsne
%type <expression> expression call literal
%type <ints> int_list
%type <strings> string_list

%start program

%%

program : tMODULE tIDENTIFIER '{' module_contents '}'                           { program = new HorseIR::Program({new HorseIR::Module($2, *$4)}); }
	;

module_contents : module_contents module_content                                { $1->push_back($2); $$ = $1; }
	       | %empty                                                         { $$ = new std::vector<HorseIR::ModuleContent *>(); }
               ;

module_content : tDEF tIDENTIFIER '(' ')' ':' type '{' statements '}'           { $$ = new HorseIR::Method($2, $6, *$8); }
	      | tIMPORT tIDENTIFIER ';'                                         { $$ = new HorseIR::Import($2); }
              ;

type : '?'                                                                      { $$ = new HorseIR::WildcardType(); }
     | int_type                                                                 { $$ = $1; }
     | tSTRING                                                                  { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Type::String); }
     | tTABLE                                                                   { $$ = new HorseIR::TableType(); }
     | tLIST '<' type '>'                                                       { $$ = new HorseIR::ListType($3); }
     ;

int_type : tI64                                                                 { $$ = new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Type::Int64); }
	 ;

statements : statements statement                                               { $1->push_back($2); $$ = $1; }
	   | %empty                                                             { $$ = new std::vector<HorseIR::Statement *>(); }
           ;

statement : tIDENTIFIER ':' type '=' expression ';'                             { $$ = new HorseIR::AssignStatement($1, $3, $5); }
          | tRETURN tIDENTIFIER ';'                                             { $$ = new HorseIR::ReturnStatement($2); }
          ;

expression : tCHECKCAST '(' expression ',' type ')'                             { $$ = new HorseIR::CastExpression($3, $5); }
	   | call                                                               { $$ = $1; }
           | literal                                                            { $$ = $1; }
           ; 

call : tFUNCTION '(' literals ')'                                               { $$ = new HorseIR::CallExpression(std::string($1), *$3); }
     ;

literals : literalsne                                                           { $$ = $1; }
         | %empty                                                               { $$ = new std::vector<HorseIR::Expression *>(); }
         ;

literalsne : literalsne ','  literal                                            { $1->push_back($3); $$ = $1; }
           | literal                                                            { $$ = new std::vector<HorseIR::Expression *>({$1}); } 
           ;

literal : tIDENTIFIER                                                           { $$ = new HorseIR::Identifier($1); }
	| tSYMBOL                                                               { $$ = new HorseIR::Symbol($1); }
	| int_list ':' int_type                                                 { $$ = new HorseIR::Literal<long>(*$1, $3); }
        | '(' int_list ')' ':' int_type                                         { $$ = new HorseIR::Literal<long>(*$2, $5); }
        | string_list ':' tSTRING                                               { $$ = new HorseIR::Literal<std::string>(*$1, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Type::String)); }
        | '(' string_list ')' ':' tSTRING                                       { $$ = new HorseIR::Literal<std::string>(*$2, new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Type::String)); }
        ;

int_list : int_list ',' tINTVAL                                                 { $1->push_back($3); $$ = $1; } 
	 | tINTVAL                                                              { $$ = new std::vector<long>({$1}); }
         ;

string_list : string_list ',' tSTRINGVAL                                        { $1->push_back(std::string($3)); $$ = $1; } 
	    | tSTRINGVAL                                                        { $$ = new std::vector<std::string>({std::string($1)}); }
            ;


%%
