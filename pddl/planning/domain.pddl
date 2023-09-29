(define (domain tavern)
; predicates are true/false statements about a set of objects
; they are considered false unless otherwise stated
(:predicates
    (party ?p) ; p is a party
    (room ?r) ; r is a room
    (size ?s) ; s is a size
    (day ?d) ; d is a day
    (has_booked_party ?p ?d) ; the party is booked for the day
    (has_booked_room ?r ?d) ; the room is booked for the day
    (has_room_key ?p ?r) ; the party has they key to the room
    (has_reservation ?p ?r ?d) ; the party has a reservation for a room and day
    (is_vacant ?r) ; the room is empty
    (is_clean ?r) ; the room is clean
    (is_size ?o ?s) ; object is a particular size
    (fits ?s1 ?s2) ; first size fits inside the second size
    (is_current_day ?d) ; the day is the current day
    (is_next_day ?d1 ?d2) ; day two follows day one
)
; actions perform change in the world state
;   parameters are the affected variables
;   precondition is the requirement for the effect to take place
;   effect is how the world state is changed

; The room is booked by the party

; If it's the current day and the room is vacant, cleaned and fits the party size
; and the party is not booked for the day

; The party is booked and recieves the room key. 
; The room is booked, occupied, and dirty
(:action drop_inec
 :parameters (?p ?r ?d ?ps ?rs)
 :precondition
    (and 
        (party ?p) (room ?r) (day ?d) (size ?ps) (size ?rs)
        (is_current_day ?d) (not (has_booked_party ?p ?d)) (is_vacant ?r) (is_clean ?r)
        (is_size ?r ?rs) (is_size ?p ?ps) (fits ?ps ?rs)
    )
 :effect
    (and 
        (has_booked_room ?r ?d) (has_booked_party ?p ?d) (has_room_key ?p ?r) 
        (not (is_vacant ?r)) (not (is_clean ?r))
    )
)
; The room is booked by the party with a reservation

; If it's the current day and the party has a reservation
; and the room is vacant and cleaned

; The party is booked and recieves the room key. 
; The room is booked, occupied, and dirty
(:action check_in
 :parameters (?p ?r ?d)
 :precondition
    (and 
        (party ?p) (room ?r) (day ?d)
        (is_current_day ?d) (has_reservation ?p ?r ?d) (is_vacant ?r) (is_clean ?r)
    )
 :effect
    (and 
        (has_booked_room ?r ?d) (has_booked_party ?p ?d) (has_room_key ?p ?r) 
        (not (is_vacant ?r)) (not (is_clean ?r))
    )
)
; The party vacate the room

; If it's the current day and the party has the room key, 
; but the room is not booked for the day

; The party returns the room key, and the room is emptied
(:action vacate
 :parameters (?r ?p ?d)
 :precondition
    (and 
        (party ?p) (room ?r) (day ?d)
        (is_current_day ?d) (has_room_key ?p ?r) (not (has_booked_room ?r ?d))
    )
 :effect
    (and 
        (not (has_room_key ?p ?r)) (is_vacant ?r)
    )
)
; The staff clean the room

; If the room is empty and dirty

; The room is cleaned
(:action clean_room
 :parameters
    (?r)
 :precondition
    (and 
        (room ?r) (is_vacant ?r) (not (is_clean ?r))
    )
 :effect
    (and 
        (is_clean ?r)
    )
)
; Take a time-step forward one day

; If day one is the current day and day two follows day one

; Day two is the current day, and day one is no longer the current day
(:action next_day
 :parameters (?d1 ?d2)
 :precondition
    (and 
        (day ?d1) (day ?d2)
        (is_current_day ?d1) (is_next_day ?d1 ?d2)
    )
 :effect
    (and 
        (is_current_day ?d2) (not (is_current_day ?d1))
    )
)
)
