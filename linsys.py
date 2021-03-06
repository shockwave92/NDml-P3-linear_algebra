from decimal import Decimal, getcontext
from copy import deepcopy

from vector import Vector
from plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)

    def solution(self):
        """
        Return the solutions of the System
        # if there is 0 = k there are no solutions
        # if there are less planes than dimensions there are infinite solutions
        # if there are more than 1 pivot variable there are infinite solutions
        # otherwise there is a single solution
        """
        try:
            return self.do_gaussian()

        except Exception as e:
            if (str(e) == self.NO_SOLUTIONS_MSG or
                str(e) == self.INF_SOLUTIONS_MSG):
                return str(e)
            #else:
            #    raise e

    def do_gaussian(self):
        rref = self.compute_rref()
        rref.raise_exception_if_contradictory_equation()
        rref.raise_exception_if_too_few_pivots()

        sol = [rref.planes[i].constant_term for i in range(rref.dimension)]
        return Vector(sol)

    def raise_exception_if_contradictory_equation(self):
        for p in self.planes:
            try:
                p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) =='No nonzero elements found':
                    constant_term = MyDecimal(p.constant_term)
                    if not constant_term.is_near_zero():
                        raise Exception(self.NO_SOLUTIONS_MSG)
                else:
                    raise e

    def raise_exception_if_too_few_pivots(self):
        pivot = self.indices_of_first_nonzero_terms_in_each_row()
        num_pivots = sum([1 if index >= 0 else 0 for index in pivot])
        var = self.dimension
        if pivot < var:
            raise Exception(self.INF_SOLUTIONS_MSG)

    def compute_triangular_form(self):
        system = deepcopy(self)
        equ = len(system)
        var = system.dimension
        j = 0
        for i in range(equ):
            while j < var:
                c = MyDecimal(system[i].normal_vector[j])
                if c.is_near_zero():
                    swap_succeeded = system.swap_with_row_below_for_nonzero_coefficient_if_able(i,j)
                    if not swap_succeeded:
                        j +=1
                        continue
                system.clear_coeefficents_below(i,j)
                j += 1
                break
        return system

    def swap_with_row_below_for_nonzero_coefficient_if_able(self, row, col):
        equ = len(self)
        for k in range(row+1, equ):
            coefficient = MyDecimal(self[k].normal_vector[col])
            if not coefficient.is_near_zero():
                self.swap_rows(row,k)
                return True
        return False

    def clear_coeefficents_below(self, row, col):
        equ = len(self)
        beta = MyDecimal(self[row].normal_vector[col])
        for k in range(row+1, equ):
            n = self[k].normal_vector
            gamma = n[col]
            alpha = -gamma / beta
            self.add_multiple_times_row_to_row(alpha, row, k)

    def compute_rref(self):
        tf = self.compute_triangular_form()
        equ = len(tf)
        leadingVar = tf.indices_of_first_nonzero_terms_in_each_row()
        for i in range(equ):
            j = leadingVar[i]
            if j < 0:
                continue
            tf.scale_row_to_make_coefficent_equal_one(i, j)
            tf.clear_coeefficents_above(i,j)
        return tf

    def scale_row_to_make_coefficent_equal_one(self, row, col):
        n = self[row].normal_vector
        beta = Decimal('1.0') / n[col]
        self.multiply_coefficient_and_row(beta,row)

    def clear_coeefficents_above(self, row, col):
        for k in range(row):
            n = self[k].normal_vector
            a = -(n[col])
            self.add_multiple_times_row_to_row(a, row, k)

    def swap_rows(self, row1, row2):
        trow = self[row1]
        self[row1] = self[row2]
        self[row2] = trow

    def multiply_coefficient_and_row(self, coefficient, row):
        n = self[row].normal_vector
        k = self[row].constant_term
        new_normal_vector = n.times_scalar(coefficient)
        new_constant_term = k*coefficient

        self[row] = Plane(normal_vector = new_normal_vector, constant_term = new_constant_term)

    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        n1 = self[row_to_add].normal_vector
        k1 = self[row_to_add].constant_term
        n2 = self[row_to_be_added_to].normal_vector
        k2 = self[row_to_be_added_to].constant_term
        new_normal_vector = n1.times_scalar(coefficient).plus(n2)
        new_constant_term = (k1*coefficient) + k2
        self[row_to_be_added_to] = Plane(normal_vector = new_normal_vector, constant_term = new_constant_term)

    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices


    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret


class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps

#print MyDecimal('1e-9').is_near_zero()
#print MyDecimal('1e-11').is_near_zero()

p1 = Plane(normal_vector=Vector(['5.862','1.178','-10.3662']), constant_term='-8.15')
p2 = Plane(normal_vector=Vector(['-2.931','-0.589','5.183']), constant_term='-4.075')
s = LinearSystem([p1,p2])
print s.solution()

p1 = Plane(normal_vector=Vector(['8.631','5.112','-1.816']), constant_term='-5.113')
p2 = Plane(normal_vector=Vector(['4.315','11.132','-5.27']), constant_term='-6.775')
p3 = Plane(normal_vector=Vector(['-2.158','3.01','-1.727']), constant_term='-0.831')
s = LinearSystem([p1,p2,p3])
print s.solution()

p1 = Plane(normal_vector=Vector(['5.262','2.739','-9.878']), constant_term='-3.441')
p2 = Plane(normal_vector=Vector(['5.111','6.358','7.638']), constant_term='-2.152')
p3 = Plane(normal_vector=Vector(['2.016','-9.924','-1.367']), constant_term='-9.278')
p4 = Plane(normal_vector=Vector(['2.167','-13.543','-18.883']), constant_term='-10.567')
s = LinearSystem([p1,p2,p3,p4])
print s.solution()
