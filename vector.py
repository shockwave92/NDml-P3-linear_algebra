import math
from decimal import Decimal, getcontext

getcontext().prec = 10

class Vector(object):
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalized zero vector'
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            self.dimension = len(self.coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def plus(self,v):
        new_coordinates = [x+y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def minus(self,v):
        new_coordinates = [x-y for x,y in zip(self.coordinates, v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        new_coordinates = [Decimal(c)*x for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        squared = [Decimal(x)*x for x in self.coordinates]
        return math.sqrt(sum(squared))

    def normalized(self):
        try:
            magnitude = self.magnitude()
            return self.times_scalar(Decimal('1.0')/Decimal(magnitude))
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

    def component_parallel(self,basis):
        try:
            u = basis.normalized()
            weight = self.dot(u)
            return u.times_scalar(weight)
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

    def component_orthogonal(self,basis):
        try:
            projection = self.component_parallel(basis)
            return self.minus(projection)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e

    def is_orthogonal(self,v,tolerance = 1e-10):
        return abs(self.dot(v)) < tolerance

    def is_parallel_to(self,v):
        #this function to check is parallel or not
        return (self.is_zero() or v.is_zero() or self.angle_with(v) == math.pi or self.angle_with(v) == 0)

    def is_zero(self,tolerance = 1e-10):
        return set(self.coordinates) == set([Decimal(0)])

    def cross(self,v):
        try:
            x1,y1,z1 = self.coordinates
            x2,y2,z2 = v.coordinates
            new_coordinates = [y1*z2 - y2*z1, -(x1*z2 - x2*z1), x1*y2 - x2*y1]
            return Vector(new_coordinates)
        except ValueError as e:
            msg = str(e)
            if msg == 'need more than 2 value to unpack':
                self_embedded_in_R3 = Vector(self.coordinates + ('0',))
                V_embedded_in_R3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_R3.cross(V_embedded_in_R3)
            elif (msg == 'too many values to unpack' or
                  msg == 'need more than one value to unpack'):
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
            else:
                raise e

    def area_of_parallelogrm(self,v):
        cross_product = self.cross(v)
        return cross_product.magnitude()

    def area_of_triangle(self,v):
        return self.area_of_parallelogrm(v) / 2.0

    def dot(self,v):
        return sum([x*y for x,y in zip(self.coordinates,v.coordinates)])

    def angle_with(self,v,in_degrees = False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = math.acos(round(u1.dot(u2),10))
            if in_degrees:
                degrees_per_radian = (180. / math.pi)
                return Decimal(angle_in_radians * degrees_per_radian)
            else:
                return Decimal(angle_in_radians)

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e
    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def __getitem__(self, i):
        return self.coordinates[i]

    def __iter__(self):
        return self.coordinates.__iter__()

'''v1 = Vector(['8.462','7.839','-8.187'])
w1 = Vector(['6.984','-5.975','4.778'])
v2 = Vector(['-8.987','-9.838','5.031'])
w2 = Vector(['-4.268','-1.861','-8.866'])
v3 = Vector(['1.5','9.547','3.691'])
w3 = Vector(['-6.007','0.124','5.772'])
#print w4.angle_with(v1, in_degrees = True)
print v1.cross(w1)
print v2.area_of_parallelogrm(w2)
print v3.area_of_triangle(w3)
'''
