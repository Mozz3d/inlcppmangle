import argparse, re
from types import SimpleNamespace


Keys = SimpleNamespace(
    OPERATOR = 'operator',
    VFTABLE = 'vftable',

    L_ANG_BRACKET = '<',
    R_ANG_BRACKET = '>',

    L_SQR_BRACKET = '[',
    R_SQR_BRACKET = ']',

    L_PAREN = '(',
    R_PAREN = ')',

    SEPARATOR = ',',

    SCOPE_RESOLUTION = '::',

    PUBLIC = 'public',
    PRIVATE = 'private',
    PROTECTED = 'protected',
    ACCESS_SCOPE = ':',

    CONST = 'const',
    VOLATILE = 'volatile',

    STATIC = 'static',
    VIRTUAL = 'virtual',
    CDECL = '__cdecl',
    STDCALL = '__stdcall',
    FASTCALL = '__fastcall',
    
    VOID = 'void',
    BOOL = 'bool',
    FLOAT = 'float',
    DOUBLE = 'double',
    SIGNED = 'signed',
    UNSIGNED = 'unsigned',
    CHAR = 'char',
    SHORT = 'short',
    INT = 'int',
    LONG = 'long',
    INT64 = '__int64',
    WCHAR = 'wchar_t',
    CLASS = 'class',
    STRUCT = 'struct',
    UNION = 'union',

    PTR = '*',
    REF = '&',
    RVAL_REF = '&&',
    PTR64 = '__ptr64',
 
    DESTRUCTOR = '~',
    ASTERISK = '*'
)

class lazyattr(property):
    def __init__(self, getter):
        self.getter = getter
        self.name = getter.__name__
        
    def __get__(self, instance, owner):
        value = self.getter(owner)
        setattr(owner, self.name, value)
        return value


class MetaNode(type):
    def __instancecheck__(cls, instance):
        if type.__instancecheck__(cls, instance):
            return True
        
        if hasattr(cls, '_flattened_producers'):
            return any(type.__instancecheck__(producer, instance) for producer in cls._flattened_producers)
        return False


class Node(metaclass=MetaNode):
    @lazyattr
    def genericPattern(cls):
        return re.sub(r'\(\?P<[^>]+>', '(?:', cls.regex.pattern)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        if hasattr(cls, 'producers') and cls.producers:
            def flatten_producers(producers):
                terminals = []
                for producer in producers:
                    if hasattr(producer, 'producers') and producer.producers:
                        if '__init__' in producer.__dict__:
                            terminals.append(producer)
                        terminals.extend(flatten_producers(producer.producers))
                    else:
                        terminals.append(producer)
                return tuple(set(terminals))
            
            cls._flattened_producers = flatten_producers(cls.producers)
    
    def __new__(cls, string: str):
        if not hasattr(cls, 'producers') or not cls.producers:
            return super().__new__(cls)
            
        for producer in cls.producers:
            try:
                return producer(string)
            except SyntaxError:
                continue
        if '__init__' in cls.__dict__:
            return super().__new__(cls)

        raise SyntaxError(f"invalid {cls.__name__} '{string}'")
    
    def parse(self, string):
        match = self.regex.fullmatch(string)
        if not match:
            raise SyntaxError(f'invalid {type(self).__name__} "{string}"')
        
        return match
    
    def __eq__(self, other):
        return str(self) == str(other)
    
    def __hash__(self):
        return hash(str(self))


class AccessSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.PUBLIC}|{Keys.PRIVATE}|{Keys.PROTECTED}', re.VERBOSE)
    
    @lazyattr
    def PUBLIC(cls):
        return cls(Keys.PUBLIC)
    
    @lazyattr
    def PRIVATE(cls):
        return cls(Keys.PRIVATE)
    
    @lazyattr
    def PROTECTED(cls):
        return cls(Keys.PROTECTED)
    
    def __init__(self, string: str):
        self.specifier: str = self.parse(string).group()

    def __str__(self) -> str:
        return self.specifier


class ResolutionSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.STATIC}|{Keys.VIRTUAL}', re.VERBOSE)
    
    @lazyattr
    def STATIC(cls):
        return cls(Keys.STATIC)
    
    @lazyattr
    def VIRTUAL(cls):
        return cls(Keys.VIRTUAL)
    
    def __init__(self, string: str):
        self.specifier: str = self.parse(string).group()
    
    def __str__(self) -> str:
        return self.specifier


class CallConvention(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.CDECL}|{Keys.STDCALL}|{Keys.FASTCALL}', re.VERBOSE)

    @lazyattr
    def CDECL(cls):
        return cls(Keys.CDECL)
    
    @lazyattr
    def STDCALL(cls):
        return cls(Keys.STDCALL)

    def __init__(self, string: str):
        self.specifier: str = self.parse(string).group()
    
    def __str__(self) -> str:
        return self.specifier


class CVQualifierSeq(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<first>{Keys.CONST}|{Keys.VOLATILE})
            (?:\s+(?P<last>{Keys.CONST}|{Keys.VOLATILE}))?
        ''', re.VERBOSE)
    
    @lazyattr
    def CONST(cls):
        return cls(Keys.CONST)
    
    @lazyattr
    def VOLATILE(cls):
        return cls(Keys.VOLATILE)
    
    @lazyattr
    def CONST_VOLATILE(cls):
        return cls.CONST + cls.VOLATILE
    
    def __init__(self, string):
        match = self.parse(str(string))
        if len(set(match.groups())) < len(match.groups()):
            raise SyntaxError(f"invalid {type(self).__name__} '{string}'")
        
        self.qualifiers: list[str] = [cvQual for cvQual in match.groups() if cvQual]

    def __iter__(self):
        for i in range(len(self.qualifiers)) :
            yield self.qualifiers[i]
    
    def __getitem__(self, key):
        return self.qualifiers[key]
    
    def __add__(self, other):
        return CVQualifierSeq(f'{self} {other}')
    
    def __str__(self) -> str:
        return ' '.join(self.qualifiers).strip()


class ConstantExpression(Node):
    @lazyattr
    def regex(cls):
        return re.compile(r'\d+', re.VERBOSE)
    
    def __init__(self, string: str):
        self.value = self.parse(string).group()
    
    def __str__(self) -> str:
        return self.value


class Identifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'[_a-zA-Z][_a-zA-Z0-9]*', re.VERBOSE)

    def __init__(self, string: str):
        self.name = self.parse(string).group()

    def __str__(self):
        return self.name


class TemplateArgsList(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.L_ANG_BRACKET}.*{Keys.R_ANG_BRACKET}', re.VERBOSE)

    @staticmethod
    def split_arguments(args_list: str):
        if not args_list:
            return []
        result: List[str] = []
        buffer: str = ""
        depth: int = 0

        if args_list.startswith(Keys.L_ANG_BRACKET) and args_list.endswith(Keys.R_ANG_BRACKET):
            args_list = args_list[1:-1]

        for char in args_list:
            if char == Keys.L_ANG_BRACKET:
                depth += 1
            elif char == Keys.R_ANG_BRACKET:
                depth = max(depth - 1, 0)
            elif char == Keys.SEPARATOR and depth == 0:
                result.append(buffer.strip())
                buffer = ""
                continue
            buffer += char

        if buffer:
            result.append(buffer.strip())

        return result

    def __init__(self, string: str):
        args = self.split_arguments(self.parse(string).group())
        self.args_list = [TemplateArgument(arg) for arg in args]

    def __len__(self):
        return len(self.args_list)

    def __getitem__(self, key):
        return self.args_list[key]

    def __str__(self):
        return (
            f'{Keys.L_ANG_BRACKET}'
            f'{f'{Keys.SEPARATOR} '.join(str(arg) for arg in self.args_list)}'
            f'{Keys.R_ANG_BRACKET}'
        )


class SimpleTemplateID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<identifier>{Identifier.genericPattern})
            (?P<argsList>{TemplateArgsList.genericPattern})
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self.identifier = Identifier(match.group('identifier'))
        self.template_args_list = TemplateArgsList(match.group('argsList'))

    def __str__(self):
        return f"{self.identifier}{self.template_args_list}"


class UnqualifiedID(Node):
    producers = (Identifier, SimpleTemplateID)
    @lazyattr
    def regex(cls):
        return re.compile(rf'(?:{Identifier.genericPattern})(?:{TemplateArgsList.genericPattern})?', re.VERBOSE)


class OverloadableOperator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(r'''
            \+\+|--|<=>|==|!=|<=|>=|{Keys.PTR}|&&|\|\||\+=|-=|\*=|/=|%=|<<=|>>=|&=|\|=|\^=|<<|>>|->*|->|\[\]|\(\)|[+\-*/%<>&|^~!=,]
        ''', re.VERBOSE)
    
    @lazyattr
    def PTR(cls):
        return cls(Keys.PTR)
    
    def __init__(self, string: str):
        self.operator: str = self.parse(string).group()

    def __str__(self):
        return self.operator


class OperatorFunctionTemplateID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<operatorFuncID>{OperatorFunctionID.genericPattern})
            (?P<templateArgsList>{TemplateArgsList.genericPattern})
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string) 
        self.operator = OverloadableOperator(match.group('operatorFuncID'))
        self.template_args_list = TemplateArgsList(match.group('templateArgsList'))

    def __str__(self):
        return f"operator{self.operator}{self.template_args_list}"


class TemplateID(Node):
    producers = (SimpleTemplateID, OperatorFunctionTemplateID)
    @lazyattr
    def regex(cls):
        return re.compile('|'.join(f'(?:{p.genericPattern})' for p in cls.producers), re.VERBOSE)


class NestedNameSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'(?:{UnqualifiedID.genericPattern}{Keys.SCOPE_RESOLUTION})+', re.VERBOSE)
    
    @staticmethod
    def rpartition_scope(scope: str):
        if not scope:
            return '', '', scope
        depth = 0

        for i in range(len(scope) - 1, 0, -1):
            char = scope[i]

            if char == Keys.R_ANG_BRACKET:
                depth += 1
            elif char == Keys.L_ANG_BRACKET:
                depth = max(depth - 1, 0)

            if depth == 0 and scope[i-1:i+1] == Keys.SCOPE_RESOLUTION:
                return scope[:i-1], Keys.SCOPE_RESOLUTION, scope[i+1:]

        return "", "", scope
    
    def __init__(self, string: str):
        previous, __, name = self.rpartition_scope(self.parse(string).group().rstrip(Keys.SCOPE_RESOLUTION))
        self.identifier = UnqualifiedID(name)
        self.scope = NestedNameSpecifier(f"{previous}{Keys.SCOPE_RESOLUTION}") if previous else None

    def __str__(self) -> str:
        return f'{self.scope or ''}{self.identifier}{Keys.SCOPE_RESOLUTION}'


class FundamentalTypeSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{Keys.VOID}|{Keys.BOOL}|{Keys.FLOAT}|{Keys.DOUBLE})
            |
            (?:(?P<signage>{Keys.SIGNED}|{Keys.UNSIGNED})\s+)?
            (?:{Keys.CHAR}|{Keys.WCHAR}|{Keys.SHORT}|{Keys.INT}|{Keys.LONG}|{Keys.INT64})
        ''', re.VERBOSE)

    @lazyattr
    def VOID(cls):
        return cls(Keys.VOID)

    @lazyattr
    def BOOL(cls):
        return cls(Keys.BOOL)

    @lazyattr
    def FLOAT(cls):
        return cls(Keys.FLOAT)

    @lazyattr
    def DOUBLE(cls):
        return cls(Keys.DOUBLE)
    
    @lazyattr
    def SCHAR(cls):
        return cls(f'{Keys.SIGNED} {Keys.CHAR}')
    
    @lazyattr
    def CHAR(cls):
        return cls(Keys.CHAR)
    
    @lazyattr
    def UCHAR(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.CHAR}')
    
    @lazyattr
    def WCHAR(cls):
        return cls(Keys.WCHAR)
    
    @lazyattr
    def SHORT(cls):
        return cls(Keys.SHORT)
    
    @lazyattr
    def USHORT(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.SHORT}')

    @lazyattr
    def INT(cls):
        return cls(Keys.INT)
    
    @lazyattr
    def UINT(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.INT}')
    
    @lazyattr
    def LONG(cls):
        return cls(Keys.LONG)
    
    @lazyattr
    def ULONG(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.LONG}')
    
    @lazyattr
    def INT64(cls):
        return cls(Keys.INT64)
    
    @lazyattr
    def UINT64(cls):
        return cls(f'{Keys.UNSIGNED} {Keys.INT64}')
    
    def __init__(self, string: str):
        match = self.parse(string)
        self.specifier = match.group()
        self.signage = match.group('signage')
        
        if Keys.CHAR not in self.specifier and self.signage == Keys.SIGNED:
            self.specifier = self.specifier.replace(Keys.SIGNED, '').strip()
    
    def __str__(self) -> str:
        return f'{self.specifier}'


class ClassKey(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'{Keys.CLASS}|{Keys.STRUCT}|{Keys.UNION}', re.VERBOSE)
    
    @lazyattr
    def CLASS(cls):
        return cls(Keys.CLASS)
    
    @lazyattr
    def STRUCT(cls):
        return cls(Keys.STRUCT)
    
    @lazyattr
    def UNION(cls):
        return cls(Keys.UNION)
    
    def __init__(self, string: str):
        self.key: str = self.parse(string).group()
    
    def __str__(self) -> str:
        return self.key


class ElaboratedTypeSpecifier(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<classKey>{ClassKey.genericPattern})\s+
            (?P<name>(?:{NestedNameSpecifier.genericPattern})?(?:{UnqualifiedID.genericPattern}))
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self.class_key = ClassKey(match.group('classKey'))
        self.type_name = IDExpression(match.group('name'))

    def __str__(self) -> str:
        return f'{self.class_key} {self.type_name}'


class TypeSpecifier(Node):
    producers = (FundamentalTypeSpecifier, ElaboratedTypeSpecifier)
    
    @lazyattr
    def regex(cls):
        return re.compile('|'.join(f'(?:{p.genericPattern})' for p in cls.producers), re.VERBOSE)


class TypeID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<typeSpecifier>{TypeSpecifier.genericPattern})
            (?:\s+(?P<cvQualSeq>{CVQualifierSeq.genericPattern}))?
            (?:\s*(?P<ptrDeclarator>{PtrAbstractDeclarator.genericPattern}))?
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self.type_spec = TypeSpecifier(match.group('typeSpecifier'))
        self.cv_qualifiers = CVQualifierSeq(match.group('cvQualSeq')) if match.group('cvQualSeq') else None
        self.ptr_declarator = (
            PtrAbstractDeclarator(match.group('ptrDeclarator')) 
            if match.group('ptrDeclarator') 
            else None
        )
        
        if self.cv_qualifiers and self.isPtr():
            if Keys.CONST in self.cv_qualifiers:
                self.ptr_declarator.isPtrToConst = True
            if Keys.VOLATILE in self.cv_qualifiers:
                self.ptr_declarator.isPtrToVolatile = True
    
    def __str__(self) -> str:
        return f'{self.type_spec} {self.cv_qualifiers or ''}{self.ptr_declarator or ''}'.strip()
    
    def isElaborated(self):
        return isinstance(self.type_spec, ElaboratedTypeSpecifier)
    
    def isFundamental(self):
        return isinstance(self.type_spec, FundamentalTypeSpecifier)
    
    def isPtr(self):
        return bool(self.ptr_declarator)


class TemplateArgument(Node):
    producers = (TypeID, ConstantExpression)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'{TypeID.genericPattern}|{ConstantExpression.genericPattern}', re.VERBOSE)


class PtrOperator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<ptrToMemberOf>{NestedNameSpecifier.genericPattern})?
            (?P<operator>{Keys.RVAL_REF}|{Keys.REF}|{re.escape(Keys.PTR)})\s*
            (?P<cvQualSeq>{CVQualifierSeq.genericPattern})?
            (?:\s+{Keys.PTR64})? # All pointers are assumed to be 64 bits
        ''', re.VERBOSE)

    @lazyattr
    def PTR(cls):
        return cls(Keys.PTR)

    @lazyattr
    def PTR_CONST(cls):
        return cls(f'{Keys.PTR} {Keys.CONST}')

    @lazyattr
    def PTR_VOLATILE(cls):
        return cls(f'{Keys.PTR} {Keys.VOLATILE}')

    @lazyattr
    def PTR_CONST_VOLATILE(cls):
        return cls(f'{Keys.PTR} {Keys.CONST} {Keys.VOLATILE}')

    @lazyattr
    def REF(cls):
        return cls(Keys.REF)

    @lazyattr
    def RVAL_REF(cls):
        return cls(Keys.RVAL_REF)

    def __init__(self, string: str):
        match = self.parse(string)
        self.ptr_to_member_of = (
            NestedNameSpecifier(match.group('ptrToMemberOf')) 
            if match.group('ptrToMemberOf') 
            else None
        )
        self.operator = match.group('operator')
        self.cv_qualifiers = (
            CVQualifierSeq(match.group('cvQualSeq')) 
            if match.group('cvQualSeq') and self.operator == Keys.PTR 
            else None
        )

    def __str__(self) -> str:
        return f'{self.ptr_to_member_of or ''}{self.operator} {self.cv_qualifiers or ''}'.strip()


class ConstructorID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
                (?P<scope>{NestedNameSpecifier.genericPattern})
                (?P<name>{UnqualifiedID.genericPattern})
            ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = UnqualifiedID(match.group('name'))
        
        if self.identifier != self.scope.identifier:
            raise SyntaxError(f"invalid {type(self).__name__} '{string}'")
    
    def __str__(self) -> str:
        return f'{self.scope}{self.identifier}'


class DestructorID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            {Keys.DESTRUCTOR}(?P<name>{UnqualifiedID.genericPattern})
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = UnqualifiedID(match.group('name'))
        
        if self.identifier != self.scope.identifier:
            raise SyntaxError(f"invalid {type(self).__name__} '{string}'")
    
    def __str__(self) -> str:
        return f'{self.scope}{Keys.DESTRUCTOR}{self.identifier}'


class OperatorFunctionID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            \b{Keys.OPERATOR}\s*\b
            (?P<overloadableOp>{OverloadableOperator.genericPattern})
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = OverloadableOperator(match.group('overloadableOp'))

    def __str__(self):
        return f"{self.scope}{Keys.OPERATOR} {self.identifier}"


class ImplicitPropertyID(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            (?P<name>`{Keys.VFTABLE}'|{Keys.VFTABLE})
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = Keys.VFTABLE
    
    def __str__(self) -> str:
        return f"{self.scope}`{self.identifier}'"


class QualifiedID(Node):
    producers = (ConstructorID, DestructorID, OperatorFunctionID, ImplicitPropertyID)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<scope>{NestedNameSpecifier.genericPattern})
            (?P<name>{UnqualifiedID.genericPattern})
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self.scope = NestedNameSpecifier(match.group('scope'))
        self.identifier = UnqualifiedID(match.group('name'))
    
    def __str__(self) -> str:
        return f'{self.scope}{self.identifier}'


class IDExpression(Node):
    producers = (UnqualifiedID, QualifiedID)
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{NestedNameSpecifier.genericPattern})?
            (?:{Keys.DESTRUCTOR})?
            {UnqualifiedID.genericPattern}
        ''', re.VERBOSE)


class PtrAbstractDeclarator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'(?:{PtrOperator.genericPattern}\s*)+', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        ptrOpMatches = PtrOperator.regex.findall(match.group())
        self.operator = PtrOperator(''.join(ptrOpMatches.pop()))
        
        prevStr = ''.join(ptrOpMatches.pop()) if ptrOpMatches else None
        self.prev = PtrAbstractDeclarator(prevStr) if prevStr else None
        
        self.isPtrToConst = False
        self.isPtrToVolatile = False
        
        if self.prev and self.prev.operator.cv_qualifiers:
            if CVQualifier.CONST in self.prev.operator.cv_qualifiers:
                self.isPtrToConst = True
            if CVQualifier.VOLATILE in self.prev.operator.cv_qualifiers:
                self.isPtrToVolatile = True

    def __str__(self) -> str:
        return f'{self.prev or ''} {self.operator}'.strip()


#class SubscriptOperator(Node):
#    @lazyattr
#    def regex(cls):
#        return re.compile(rf'\{Keys.L_SQR_BRACKET}\s*(?P<constLen>\d*)\s*\{Keys.R_SQR_BRACKET}', re.VERBOSE)
#    
#    def __init__(self, string: str):
#        self.length = match.group('constLen')
#    
#    def __str__(self) -> str:
#        return f'{Keys.L_SQR_BRACKET}{self.length}{Keys.R_SQR_BRACKET}'
#
#
#class ArrAbstractDeclarator(Node):
#    @lazyattr
#    def regex(cls):
#        return re.compile(rf'''
#            (?P<name>{TypeSpecifier.genericPattern})\s*
#            (?P<operator>{SubscriptOperator.genericPattern})
#        ''', re.VERBOSE)
#    
#    def __init__(self, string: str):
#        self.type_spec = TypeSpecifier(match.group('name'))
#        self.operator = SubscriptOperator(match.group('operator'))
#    
#    def __str__(self) -> str:
#        return f'{self.type_spec}{self.operator}'
#
#
#class NoptrAbstractDeclarator(Node):
#    producers = (ArrAbstractDeclarator) #, ParametersDeclarator
#    
#    @lazyattr
#    def regex(cls):
#        return re.compile('|'.join(f'(?:{p.genericPattern})' for p in NoptrAbstractDeclarator.producers), re.VERBOSE)
#
#
#class AbstractDeclarator:
#    producers = (NoptrAbstractDeclarator, PtrAbstractDeclarator)
#    
#    @lazyattr
#    def regex(cls):
#        return re.compile('|'.join(f'(?:{p.genericPattern})' for p in AbstractDeclarator.producers), re.VERBOSE)


class FunctionClass(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:
                (?P<access>{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?P<res>{ResolutionSpecifier.genericPattern})?
            )?
        ''', re.VERBOSE)
    
    @lazyattr
    def GLOBAL(cls):
        return cls('')
    
    @lazyattr
    def PRIVATE(cls):
        return cls(Keys.PRIVATE)
    
    @lazyattr
    def PROTECTED(cls):
        return cls(Keys.PROTECTED)
    
    @lazyattr
    def PUBLIC(cls):
        return cls(Keys.PUBLIC)
    
    @lazyattr
    def PRIVATE_STATIC(cls):
        return cls(Keys.PRIVATE + Keys.STATIC)
    
    @lazyattr
    def PROTECTED_STATIC(cls):
        return cls(Keys.PROTECTED + Keys.STATIC)
    
    @lazyattr
    def PUBLIC_STATIC(cls):
        return cls(Keys.PUBLIC + Keys.STATIC)
    
    @lazyattr
    def PRIVATE_VIRTUAL(cls):
        return cls(Keys.PRIVATE + Keys.VIRTUAL)
    
    @lazyattr
    def PROTECTED_VIRTUAL(cls):
        return cls(Keys.PROTECTED + Keys.VIRTUAL)
    
    @lazyattr
    def PUBLIC_VIRTUAL(cls):
        return cls(Keys.PUBLIC + Keys.VIRTUAL)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self.access = AccessSpecifier(match.group('access')) if match.group('access') else None
        self.resolution = (
            ResolutionSpecifier(match.group('res')) if match.group('res') and self.access 
            else None
        )
    
    def __str__(self) -> str:
        return (
            f'{self.access or ''}{':' if self.access else ''}'
            f'{self.resolution if self.access and self.resolution else ''}'
        ).strip()


class ParametersDeclarator(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            {re.escape(Keys.L_PAREN)}
            (?P<paramsList>
                (?:{TypeID.genericPattern})(?:\s*{Keys.SEPARATOR}\s*(?:{TypeID.genericPattern}))*
            )?
            {re.escape(Keys.R_PAREN)}
        ''', re.VERBOSE)
    
    @staticmethod
    def split_parameters(params_list: str):
        if not params_list:
            return []
        result: List[str] = []
        buffer: str = ""
        depth: int = 0
        
        params_list = params_list.lstrip(Keys.L_PAREN)
        params_list = params_list.rstrip(Keys.R_PAREN)

        for char in params_list:
            if char in (Keys.L_ANG_BRACKET, Keys.L_PAREN):
                depth += 1
            elif char in (Keys.R_ANG_BRACKET, Keys.R_PAREN):
                depth = max(depth - 1, 0)
            elif char == Keys.SEPARATOR and depth == 0:
                result.append(buffer.strip())
                buffer = ""
                continue

            buffer += char

        if buffer:
            result.append(buffer.strip())

        return result
    
    @lazyattr
    def VOID(cls):
        return cls(f'{Keys.L_PAREN}{Keys.VOID}{Keys.R_PAREN}')
    
    def __init__(self, string: str):
        params = self.split_parameters(self.parse(string).group())
        self.params_list = [TypeID(param) for param in params or [Keys.VOID]]
    
    def __getitem__(self, key):
        return self.params_list[key]
    
    def __bool__(self):
        return self != self.VOID
    
    def __str__(self) -> str:
        return (
            f'{Keys.L_PAREN}'
            f'{Keys.SEPARATOR.join([str(param) for param in self.params_list]).strip()}'
            f'{Keys.R_PAREN}'
        )


class FuncNode(Node):
    def __str__(self) -> str:
        return (
            f'{self._class if self._class is not FunctionClass.GLOBAL else ''} '
            f'{self.return_type or '\b'} {self.call_conv} '
            f'{self.identifier}{self.params}'
            f'{self.instance_quals or ''}'
        ).strip()
    
    def isGlobal(self):
        return self._class == FunctionClass.GLOBAL
  
    def isStatic(self):
        return self._class.resolution == ResolutionSpecifier.STATIC
    
    def isReturnByValue(self):
        return (
            self.return_type 
            and self.return_type.isElaborated()
            and not self.return_type.isPtr()
        )


class ConstructorDefinition(FuncNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
            )\s+
            (?!(?P<retType>{TypeID.genericPattern})\s+)
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{ConstructorID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?!\s*(?P<instanceQuals>{CVQualifierSeq.genericPattern}))?
            (?:\s+{Keys.PTR64})?
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type =  None
        self.call_conv = (
            CallConvention(match.group('callConv')) 
            if match.group('callConv') 
            else CallConvention.CDECL
        )
        self.identifier = ConstructorID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_quals = None
        
        if self.identifier.identifier != self.identifier.scope.identifier:
            raise SyntaxError(f"Invalid {type(self).__name__} '{string}'")


class DestructorDefinition(FuncNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?:{Keys.VIRTUAL})?
            )\s+
            (?!(?P<retType>{TypeID.genericPattern})\s+)
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{DestructorID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?!\s*(?P<instanceQuals>{CVQualifierSeq.genericPattern}))?
            (?:\s+{Keys.PTR64})?
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type =  None
        self.call_conv = (
            CallConvention(match.group('callConv')) 
            if match.group('callConv') 
            else CallConvention.CDECL
        )
        self.identifier = DestructorID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_quals = None

        if (
            self.identifier.scope.identifier != self.identifier.identifier
            or
            self.params
        ):
            raise SyntaxError(f"Invalid {type(self).__name__} '{string}'")


class OperatorFunctionDefinition(FuncNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
            )\s+
            (?P<retType>{TypeID.genericPattern})\s+
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{OperatorFunctionID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?:\s*(?P<instanceQuals>{CVQualifierSeq.genericPattern}))?
            (?:\s+{Keys.PTR64})?
        ''', re.VERBOSE)

    def __init__(self, string: str):
        match = self.parse(string)
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type = TypeID(match.group('retType'))
        self.call_conv = (
            CallConvention(match.group('callConv')) 
            if match.group('callConv') 
            else CallConvention.CDECL
        )
        self.identifier = OperatorFunctionID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_quals = (
            CVQualifierSeq(match.group('instanceQuals')) 
            if match.group('instanceQuals') 
            else None
        )


class MethodDefinition(FuncNode):
    producers = (ConstructorDefinition, DestructorDefinition, OperatorFunctionDefinition)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<funcClass>
                (?:{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?:{ResolutionSpecifier.genericPattern})?
            )\s+
            (?P<retType>{TypeID.genericPattern})\s+
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{QualifiedID.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
            (?:\s*(?P<instanceQuals>{CVQualifierSeq.genericPattern}))?
            (?:\s+{Keys.PTR64})?
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self._class = FunctionClass(match.group('funcClass'))
        self.return_type = TypeID(match.group('retType'))
        self.call_conv = (
            CallConvention(match.group('callConv')) 
            if match.group('callConv') 
            else CallConvention.CDECL
        )
        self.identifier = QualifiedID(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_quals = (
            CVQualifierSeq(match.group('instanceQuals')) 
            if match.group('instanceQuals')
            else None
        )
        
        if self.isStatic() and self.instance_quals:
            raise RuntimeError(f"Static method cannot have instance qualifiers '{self.instance_quals}'")


class FunctionDefinition(FuncNode):
    producers = (MethodDefinition,)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<retType>{TypeID.genericPattern})\s+
            (?:(?P<callConv>{CallConvention.genericPattern})\s+)?
            (?P<identifier>{IDExpression.genericPattern})\s*
            (?P<params>{ParametersDeclarator.genericPattern})
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self._class = FunctionClass.GLOBAL
        self.return_type = TypeID(match.group('retType'))
        self.call_conv = (
            CallConvention(match.group('callConv')) 
            if match.group('callConv') 
            else CallConvention.CDECL
        )
        self.identifier = IDExpression(match.group('identifier'))
        self.params = ParametersDeclarator(match.group('params'))
        self.instance_quals = None


class VariableClass(Node):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:
                (?P<access>{AccessSpecifier.genericPattern})\s*
                (?:\{Keys.ACCESS_SCOPE}\s*)?
                (?P<resolution>{Keys.STATIC})?
            )?
        ''', re.VERBOSE)
    
    @lazyattr
    def GLOBAL(cls):
        return cls('')
    
    @lazyattr
    def PRIVATE_STATIC(cls):
        return cls(Keys.PRIVATE + Keys.STATIC)
    
    @lazyattr
    def PROTECTED_STATIC(cls):
        return cls(Keys.PROTECTED + Keys.STATIC)
    
    @lazyattr
    def PUBLIC_STATIC(cls):
        return cls(Keys.PUBLIC + Keys.STATIC)
 
    def __init__(self, string: str):
        match = self.parse(string)
        self.access = (
            AccessSpecifier(match.group('access')) 
            if match.group('access') 
            else None
        )
        self.resolution = (
            ResolutionSpecifier(match.group('resolution')) 
            if match.group('resolution') and self.access 
            else None
        )
    
    def __str__(self) -> str:
        return (
            f'{self.access or ''}{':' if self.access else ''} '
            f'{self.resolution if self.access and self.resolution else ''}'
        ).strip()


class VarNode(Node):
    def __str__(self) -> str:
        return (
            f'{self._class if self._class and self._class is not VariableClass.GLOBAL else ''} '
            f'{self.decl_type or ''} {self.identifier}'
        ).strip()
    
    def isGlobal(self):
        return self._class == VariableClass.GLOBAL
  
    def isStatic(self):
        return self._class.resolution == ResolutionSpecifier.STATIC


class ImplicitPropertyDefinition(VarNode):
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?:{Keys.CONST}\s+)?
            (?P<identifier>{ImplicitPropertyID.genericPattern})
        ''', re.VERBOSE)
        
    def __init__(self, string: str):
        match = self.parse(string)
        self._class = None
        self.decl_type = None
        self.identifier = ImplicitPropertyID(match.group('identifier'))
        self.storage_quals = CVQualifierSeq.CONST
    
    def __str__(self) -> str:
        return f'{Keys.CONST} {self.identifier}'


class PropertyDefinition(VarNode):
    producers = (ImplicitPropertyDefinition,)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<varClass>{VariableClass.genericPattern})\s+
            (?P<declType>{TypeID.genericPattern})\s*
            (?P<identifier>
                {NestedNameSpecifier.genericPattern}
                {Identifier.genericPattern}
            )
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self._class = VariableClass(match.group('varClass'))
        self.decl_type = TypeID(match.group('declType'))
        self.identifier = QualifiedID(match.group('identifier'))
        self.storage_quals = (
            self.decl_type.ptr_declarator.operator.cv_qualifiers 
            if self.decl_type.ptr_declarator
            else self.decl_type.cv_qualifiers
        )


class VariableDefinition(VarNode):
    producers = (PropertyDefinition,)
    
    @lazyattr
    def regex(cls):
        return re.compile(rf'''
            (?P<declType>{TypeID.genericPattern})\s*
            (?P<identifier>
                (?:{NestedNameSpecifier.genericPattern})?
                {Identifier.genericPattern}
            )
        ''', re.VERBOSE)
    
    def __init__(self, string: str):
        match = self.parse(string)
        self._class = VariableClass.GLOBAL
        self.decl_type = TypeID(match.group('declType'))
        self.storage_quals = (
            self.decl_type.ptr_declarator.operator.cv_qualifiers 
            if self.decl_type.ptr_declarator
            else self.decl_type.cv_qualifiers
        )
        self.identifier = IDExpression(match.group('identifier')) 


class Definition(Node):
    producers = (FunctionDefinition, VariableDefinition)
    @lazyattr
    def regex(cls):
        return re.compile('|'.join(f'(?:{p.genericPattern})' for p in cls.producers), re.VERBOSE)


class Mangler:
    def __init__(self, _def=None):
        self.name_back_refs = []
        self.result = ''
        if not _def:
            return
        
        if isinstance(_def, str):
            self.original = Definition(_def)
        else:
            self.original = _def
        
        self.result = self.mangle(self.original)
    
    def mangleFunctionClass(self, func_cls: FunctionClass):
        match func_cls:
            case FunctionClass.PRIVATE:
                return 'A'
            case FunctionClass.PRIVATE_STATIC:
                return 'C'
            case FunctionClass.PRIVATE_VIRTUAL:
                return 'E'
            case FunctionClass.PROTECTED:
                return 'I'
            case FunctionClass.PROTECTED_STATIC:
                return 'K'
            case FunctionClass.PROTECTED_VIRTUAL:
                return 'M'
            case FunctionClass.PUBLIC:
                return 'Q'
            case FunctionClass.PUBLIC_STATIC:
                return 'S'
            case FunctionClass.PUBLIC_VIRTUAL:
                return 'U'
            case FunctionClass.GLOBAL:
                return 'Y'

    def mangleUnqualifiedID(self, _id: UnqualifiedID):
        if _id in self.name_back_refs:
            return str(self.name_back_refs.index(_id))
        if len(self.name_back_refs) < 10:
            self.name_back_refs.append(_id)

        match _id:
            case TemplateID():
                return self.mangleTemplateID(_id)
            case _:
                return str(_id) + '@'
    
    def mangleTemplateID(self, template_id: TemplateID):
        result = '?$'
        mangler = Mangler()
        result += mangler.mangleUnqualifiedID(template_id.identifier)

        for arg in template_id.template_args_list:
            result += mangler.mangleTemplateArg(arg)

        result += '@'
        
        return result
    
    def mangleScope(self, scope: NestedNameSpecifier):
        result = self.mangleUnqualifiedID(scope.identifier)
        if scope.scope:
            result += self.mangleScope(scope.scope)

        return result
    
    def mangleID(self, _id: IDExpression):
        match _id:
            case UnqualifiedID():
                result = self.mangleUnqualifiedID(_id)
            case ConstructorID():
                result = '?0'
            case DestructorID():
                result = '?1'
            case OperatorFunctionID():
                match _id.identifier:
                    case OverloadableOperator.PTR:
                        result = '?D'
            case ImplicitPropertyID():
                result = '?_7'
            case _:
                result = self.mangleUnqualifiedID(_id.identifier)

        if hasattr(_id, 'scope'):
            result += self.mangleScope(_id.scope)
        return result + '@'

    def mangleFundamentalType(self, fundamental_type: FundamentalTypeSpecifier):
        match fundamental_type:
            case FundamentalTypeSpecifier.VOID:
                return 'X'
            case FundamentalTypeSpecifier.SCHAR:
                return 'C'
            case FundamentalTypeSpecifier.CHAR:
                return 'D'
            case FundamentalTypeSpecifier.UCHAR:
                return 'E'
            case FundamentalTypeSpecifier.SHORT:
                return 'F'
            case FundamentalTypeSpecifier.USHORT:
                return 'G'
            case FundamentalTypeSpecifier.INT:
                return 'H'
            case FundamentalTypeSpecifier.UINT:
                return 'I'
            case FundamentalTypeSpecifier.LONG:
                return 'J'
            case FundamentalTypeSpecifier.ULONG:
                return 'K'
            case FundamentalTypeSpecifier.FLOAT:
                return 'M'
            case FundamentalTypeSpecifier.DOUBLE:
                return 'N'
            case FundamentalTypeSpecifier.INT64:
                return '_J'
            case FundamentalTypeSpecifier.UINT64:
                return '_K'
            case FundamentalTypeSpecifier.BOOL:
                return '_N'
            case FundamentalTypeSpecifier.WCHAR:
                return '_W'

    def mangleClassKey(self, cls_key: ClassKey):
        match cls_key:
            case ClassKey.CLASS:
                return 'V'
            case ClassKey.STRUCT:
                return 'U'

    def mangleElaboratedType(self, elaborated_type: ElaboratedTypeSpecifier):
        result = self.mangleClassKey(elaborated_type.class_key)
        result += self.mangleID(elaborated_type.type_name)
        return result

    def mangleCVQualifiers(self, cv_quals: CVQualifierSeq):
        match cv_quals:
            case None:
                return 'A'
            case CVQualifierSeq.CONST:
                return 'B'
            case CVQualifierSeq.VOLATILE:
                return 'C'
            case CVQualifierSeq.CONST_VOLATILE:
                return 'D'

    def mangleCallConvention(self, call_conv: CallConvention):
        match call_conv:
            case CallConvention.CDECL:
                return 'A'
            case CallConvention.STDCALL:
                return 'G'
            case CallConvention.FASTCALL:
                return 'I'

    def mangleTypeSpec(self, type_spec: TypeSpecifier):
        match type_spec:
            case FundamentalTypeSpecifier():
                return self.mangleFundamentalType(type_spec)
            case ElaboratedTypeSpecifier():
                return self.mangleElaboratedType(type_spec)

    def manglePtrOperator(self, ptr_op: PtrOperator):
        match ptr_op:
            case PtrOperator.PTR:
                return 'PE'
            case PtrOperator.PTR_CONST:
                return 'QE'
            case PtrOperator.PTR_VOLATILE:
                return 'RE'
            case PtrOperator.PTR_CONST_VOLATILE:
                return 'SE'
            case PtrOperator.REF:
                return 'AE'
            case PtrOperator.RVAL_REF:
                return '$$QE'

    def manglePtrDeclarator(self, ptr_decl: PtrAbstractDeclarator):
        result = self.manglePtrOperator(ptr_decl.operator)
        result += self.mangleCVQualifiers(
            CVQualifierSeq.CONST_VOLATILE if ptr_decl.isPtrToConst and ptr_decl.isPtrToVolatile
            else 
            CVQualifierSeq.CONST if ptr_decl.isPtrToConst
            else
            CVQualifierSeq.VOLATILE if ptr_decl.isPtrToVolatile
            else
            None
        )
        
        if ptr_decl.prev:
            result += self.manglePtrDeclarator(ptr_decl.prev)
        
        return result

    def mangleTypeID(self, type_ID: TypeID):
        if not type_ID:
            return '@'

        result = self.manglePtrDeclarator(type_ID.ptr_declarator) if type_ID.ptr_declarator else ''
        result += self.mangleTypeSpec(type_ID.type_spec)
        return result
    
    def mangleConstExpression(self, const_expr: ConstantExpression):
        result = '$0'
        value = int(const_expr.value)
        
        if value == 0:
            return result + "A@"
        elif 1 <= value <= 10:
            return result + str(value - 1)
        else:
            encoded = []
            while value != 0:
                nibble = value & 0xf
                encoded.append(chr(ord('A') + nibble))
                value >>= 4
           
            encoded.reverse()
            return result + ''.join(encoded) + '@'
    
    def mangleTemplateArg(self, template_arg: TemplateArgument):
        match template_arg:
            case TypeID():
                return self.mangleTypeID(template_arg)
            case ConstantExpression():
                return self.mangleConstExpression(template_arg)

    def mangleFunctionDef(self, func_def: FunctionDefinition):
        result = self.mangleID(func_def.identifier)
        result += self.mangleFunctionClass(func_def._class)

        if not (func_def.isStatic() or func_def.isGlobal()):
            result += 'E'
            result += self.mangleCVQualifiers(func_def.instance_quals)

        result += self.mangleCallConvention(func_def.call_conv)
        
        if func_def.isReturnByValue():
            result += '?' + self.mangleCVQualifiers(func_def.return_type.cv_qualifiers)

        result += self.mangleTypeID(func_def.return_type)
        
        for param in func_def.params:
            result += self.mangleTypeID(param)
        
        if func_def.params:
            result += '@'
        
        result += 'Z'
        return result

    def mangleVariableClass(self, var_cls: VariableClass):
        match var_cls:
            case VariableClass.PRIVATE_STATIC:
                return '0'
            case VariableClass.PROTECTED_STATIC:
                return '1'
            case VariableClass.PUBLIC_STATIC:
                return '2'
            case VariableClass.GLOBAL:
                return '3'
            case None:
                return '6'
    
    def mangleVariableDef(self, var_def: VariableDefinition):
        result = self.mangleID(var_def.identifier)
        result += self.mangleVariableClass(var_def._class)

        if var_def.decl_type:
            result += self.mangleTypeID(var_def.decl_type)
            result += 'E'
        
        result += self.mangleCVQualifiers(var_def.storage_quals)

        if var_def.decl_type is None:
            result += '@'
        
        return result
    
    def mangle(self, _def: Definition):
        result = '?'
        match _def:
            case FunctionDefinition():
                result += self.mangleFunctionDef(_def)
            case VariableDefinition():
                result += self.mangleVariableDef(_def)
        
        return result
    
    def __str__(self):
        return self.result


parser = argparse.ArgumentParser(description="Accepts inline C++ definitions for mangling and hashing")
parser.add_argument("definitions", nargs='+', default="", help='One or more quote encased inline C++ definition, e.g. "public: void MyClass::myMethod(void*) const"')

def main():
    args = parser.parse_args()
    for raw_def in args.definitions:
        _def = Definition(raw_def)
        mangled = Mangler(_def)
        print('')
        print(mangled)

if __name__ == "__main__":
    main()
